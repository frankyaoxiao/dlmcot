"""
MMaDA inference using the original repository approach
Based on Gen-Verse/MMaDA-8B-MixCoT
"""

import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer
from models import MMadaModelLM
import warnings
import os
import sys
import threading
import multiprocessing

warnings.filterwarnings("ignore")

# Process-level singleton pattern
_model = None
_tokenizer = None
_device = None
_lock = threading.Lock()
_process_id = None

def _get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def _get_process_id():
    """Get current process ID for singleton management."""
    return multiprocessing.current_process().pid

def _load_model():
    """Load the MMaDA model and tokenizer if not already loaded."""
    global _model, _tokenizer, _device, _process_id
    
    current_process_id = _get_process_id()
    
    # Check if we need to reload for a different process
    if _process_id != current_process_id:
        _model = None
        _tokenizer = None
        _device = None
        _process_id = current_process_id
    
    if _model is not None and _tokenizer is not None:
        return
    
    with _lock:
        # Double-check after acquiring lock
        if _model is not None and _tokenizer is not None:
            return
        
        _device = _get_device()
        print(f"Loading MMaDA model on {_device} (process {current_process_id})...")
        
        try:
            # Clear GPU memory before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Load tokenizer first
            print("Loading tokenizer...")
            _tokenizer = AutoTokenizer.from_pretrained(
                "Gen-Verse/MMaDA-8B-MixCoT", 
                trust_remote_code=True
            )
            print("✓ Tokenizer loaded successfully")
            
            # Set chat template
            _tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n' }}"
            
            # Load model from Hugging Face
            print("Loading model...")
            _model = MMadaModelLM.from_pretrained(
                "Gen-Verse/MMaDA-8B-MixCoT", 
                trust_remote_code=True, 
                torch_dtype=torch.bfloat16 if _device != "cpu" else torch.float32
            ).to(_device).eval()
            print("✓ Model loaded successfully")
            
            print(f"MMaDA model loaded successfully! (process {current_process_id})")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            # Reset globals on error
            _model = None
            _tokenizer = None
            _device = None
            raise

def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    """
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

@torch.no_grad()
def mmada_generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
                   cfg_scale=0., remasking='low_confidence', mask_id=126336, attention_mask=None, save_history=False):
    """
    Generate text using MMaDA's masked diffusion process.
    
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (B, L), where B is batch size.
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
        attention_mask: Optional attention mask
        save_history: Whether to save generation history for analysis.
    
    Returns:
        If save_history=False: generated sequence tensor
        If save_history=True: (generated sequence tensor, generation history dict)
    """
    if attention_mask is not None and 0.0 in attention_mask:
        attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
    else:
        attention_bias = None
    
    batch_size = prompt.shape[0]
    x = torch.full((batch_size, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)
    
    # Initialize history tracking
    history = {
        'states': [],           
        'predictions': [],      
        'finalized_at_step': {},  
        'step_info': [],        
        'mask_id': mask_id,
        'prompt_length': prompt.shape[1]
    } if save_history else None

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    global_step = 0
    
    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        
        for i in range(steps_per_block):
            if save_history:
                history['states'].append(x.clone())
                history['step_info'].append({
                    'global_step': global_step,
                    'block': num_block,
                    'block_step': i,
                    'block_start': block_start,
                    'block_end': block_end
                })
            
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_bias=attention_bias).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            
            if save_history:
                history['predictions'].append(x0.clone())

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, block_end:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            
            if save_history:
                newly_finalized = transfer_index[0].cpu().numpy()
                for pos_idx, is_finalized in enumerate(newly_finalized):
                    if is_finalized and pos_idx not in history['finalized_at_step']:
                        history['finalized_at_step'][pos_idx] = global_step
            
            x[transfer_index] = x0[transfer_index]
            global_step += 1

    if save_history:
        history['states'].append(x.clone())

    return (x, history) if save_history else x

def generate_text(prompt, gen_length=128, steps=128, block_length=32, temperature=1.0, 
                  cfg_scale=0.0, remasking='low_confidence', save_history=False, use_chat_template=True, enable_cot=False):
    """Generate text using MMaDA model."""
    _load_model()
    
    # Ensure tokenizer is loaded
    if _tokenizer is None:
        raise RuntimeError("Tokenizer not loaded properly")
    
    # Apply Chain-of-Thought system prompt if enabled
    if enable_cot:
        # Check if this is a GPQA-style question requiring "ANSWER: X" format
        if "ANSWER:" in prompt and "multiple choice" in prompt.lower():
            cot_system_prompt = "You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here. IMPORTANT: After your reasoning, you MUST provide the answer in the exact format specified in the question (e.g., 'ANSWER: A' for multiple choice questions).\n"

            '''
            cot_system_prompt = """You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here. 

CRITICAL FORMAT REQUIREMENT: After your reasoning, you MUST provide the answer in the exact format "ANSWER: X" where X is the letter (A, B, C, or D) of your chosen answer. Do NOT include any other text after the answer. The format must be exactly "ANSWER: X" with no additional characters or explanations.

Example format:
<think>
Your detailed reasoning here...
</think>
ANSWER: A
'''
        else:
            cot_system_prompt = "You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here\n"
        prompt = cot_system_prompt + prompt
    
    if use_chat_template:
        # Use chat template for conversational prompts
        try:
            messages = [{"role": "user", "content": prompt}]
            prompt = _tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        except Exception as e:
            print(f"Warning: Chat template failed, using raw prompt: {e}")
            # Fall back to raw prompt if chat template fails
    
    try:
        inputs = _tokenizer(text=prompt, return_tensors="pt", padding=True, padding_side="left")
        input_ids = inputs["input_ids"].to(_device)
    except Exception as e:
        print(f"Error in tokenization: {e}")
        raise RuntimeError(f"Tokenization failed: {e}")
    
    if save_history:
        generated_ids, history = mmada_generate(
            _model, input_ids, steps=steps, gen_length=gen_length, 
            block_length=block_length, temperature=temperature,
            cfg_scale=cfg_scale, remasking=remasking, save_history=True
        )
        generated_text = _tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        return generated_text, history, _tokenizer
    else:
        generated_ids = mmada_generate(
            _model, input_ids, steps=steps, gen_length=gen_length,
            block_length=block_length, temperature=temperature,
            cfg_scale=cfg_scale, remasking=remasking, save_history=False
        )
        generated_text = _tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        return generated_text

def _unload_model():
    """Unload the model to free memory."""
    global _model, _tokenizer, _device
    
    if _model is not None:
        del _model
        _model = None
        
    if _tokenizer is not None:
        del _tokenizer
        _tokenizer = None
        
    _device = None
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("MMaDA model unloaded successfully!")

def generate_texts_batch(prompts, gen_length=128, steps=128, block_length=32, temperature=0.0, 
                        cfg_scale=0.0, remasking='low_confidence', save_history=False, 
                        batch_size=None, unload_after=False, use_chat_template=False, enable_cot=False):
    """Generate text for multiple prompts in batches."""
    _load_model()
    
    if batch_size is None:
        batch_size = len(prompts)
    
    results = []
    
    try:
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_results = []
            
            for prompt in batch_prompts:
                result = generate_text(
                    prompt, gen_length=gen_length, steps=steps,
                    block_length=block_length, temperature=temperature,
                    cfg_scale=cfg_scale, remasking=remasking, save_history=save_history,
                    use_chat_template=use_chat_template, enable_cot=enable_cot
                )
                batch_results.append(result)
            
            results.extend(batch_results)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    finally:
        if unload_after:
            _unload_model()
    
    return results
