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
                   cfg_scale=0., remasking='low_confidence', mask_id=126336, attention_mask=None, save_history=False, seed=None):
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
        seed: Random seed for deterministic generation. If None, generation will be random.
    
    Returns:
        If save_history=False: generated sequence tensor
        If save_history=True: (generated sequence tensor, generation history dict)
    """
    # Set random seed for deterministic generation if specified
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # Set deterministic mode for CUDA operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    if attention_mask is not None and 0.0 in attention_mask:
        attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
    else:
        attention_bias = None
    
    batch_size = prompt.shape[0]
    x = torch.full((batch_size, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)
    
    # Initialize history tracking with enhanced information
    history = {
        'states': [],           # Complete token states at each step
        'predictions': [],      # Model predictions at each step
        'finalized_at_step': {},  # When each token position was finalized
        'step_info': [],        # Metadata about each step
        'mask_id': mask_id,
        'prompt_length': prompt.shape[1],
        'generation_params': {
            'steps': steps,
            'gen_length': gen_length,
            'block_length': block_length,
            'temperature': temperature,
            'cfg_scale': cfg_scale,
            'remasking': remasking
        },
        'block_structure': {
            'num_blocks': gen_length // block_length,
            'steps_per_block': steps // (gen_length // block_length)
        },
        'token_confidence': [],  # Confidence scores for each prediction
        'remasking_decisions': [],  # Which tokens were remasked at each step
        'generation_progress': []  # Progress tracking for each block
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
        
        if save_history:
            history['generation_progress'].append({
                'block': num_block,
                'block_start': block_start,
                'block_end': block_end,
                'masked_tokens_in_block': block_mask_index.sum().item(),
                'steps_for_block': steps_per_block
            })
        
        for i in range(steps_per_block):
            if save_history:
                # Save complete state
                history['states'].append(x.clone())
                history['step_info'].append({
                    'global_step': global_step,
                    'block': num_block,
                    'block_step': i,
                    'block_start': block_start,
                    'block_end': block_end,
                    'step_type': 'diffusion_step'
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
                
                # Save confidence scores for predictions
                if remasking == 'low_confidence':
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                    history['token_confidence'].append(x0_p.clone())
                else:
                    # For random remasking, we don't have confidence scores
                    history['token_confidence'].append(None)

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
                # Track which tokens were finalized at this step
                newly_finalized = transfer_index[0].cpu().numpy()
                for pos_idx, is_finalized in enumerate(newly_finalized):
                    if is_finalized and pos_idx not in history['finalized_at_step']:
                        history['finalized_at_step'][pos_idx] = global_step
                
                # Track remasking decisions
                history['remasking_decisions'].append({
                    'step': global_step,
                    'tokens_finalized': transfer_index.sum().item(),
                    'finalized_positions': transfer_index[0].nonzero(as_tuple=True)[0].cpu().tolist() if transfer_index[0].any() else [],
                    'confidence_threshold': num_transfer_tokens[0, i].item()
                })
            
            x[transfer_index] = x0[transfer_index]
            global_step += 1

    if save_history:
        # Save final state
        history['states'].append(x.clone())
        
        # Add summary statistics
        history['summary'] = {
            'total_steps': global_step,
            'total_states_saved': len(history['states']),
            'total_predictions_saved': len(history['predictions']),
            'finalized_tokens_count': len(history['finalized_at_step']),
            'generation_completed': True
        }

    return (x, history) if save_history else x

def save_generation_history_to_file(history, tokenizer, output_file=None):
    """
    Save the complete generation history to a detailed text file.
    
    Args:
        history: The generation history dictionary from mmada_generate
        tokenizer: The tokenizer used for generation
        output_file: Optional output file path. If None, creates a temporary file.
    
    Returns:
        str: Path to the saved file
    """
    import tempfile
    import os
    from datetime import datetime
    
    if output_file is None:
        # Create temporary file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_dir = tempfile.gettempdir()
        output_file = os.path.join(temp_dir, f"mmada_generation_history_{timestamp}.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MMaDA GENERATION HISTORY REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output file: {output_file}\n\n")
        
        # Generation Parameters
        f.write("GENERATION PARAMETERS:\n")
        f.write("-" * 40 + "\n")
        params = history['generation_params']
        f.write(f"Steps: {params['steps']}\n")
        f.write(f"Generation Length: {params['gen_length']}\n")
        f.write(f"Block Length: {params['block_length']}\n")
        f.write(f"Temperature: {params['temperature']}\n")
        f.write(f"CFG Scale: {params['cfg_scale']}\n")
        f.write(f"Remasking Strategy: {params['remasking']}\n\n")
        
        # Show the complete prompt
        f.write("PROMPT:\n")
        f.write("-" * 40 + "\n")
        prompt_tokens = history['states'][0][0].cpu().tolist()[:history['prompt_length']]
        prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
        f.write(f"{repr(prompt_text)}\n\n")
        
        # Block Structure
        f.write("BLOCK STRUCTURE:\n")
        f.write("-" * 40 + "\n")
        block_info = history['block_structure']
        f.write(f"Number of Blocks: {block_info['num_blocks']}\n")
        f.write(f"Steps per Block: {block_info['steps_per_block']}\n\n")
        
        # Generation Progress
        f.write("GENERATION PROGRESS BY BLOCK:\n")
        f.write("-" * 40 + "\n")
        for progress in history['generation_progress']:
            f.write(f"Block {progress['block']}: Positions {progress['block_start']}-{progress['block_end']-1}, "
                   f"Masked Tokens: {progress['masked_tokens_in_block']}, Steps: {progress['steps_for_block']}\n")
        f.write("\n")
        
        # Summary Statistics
        f.write("SUMMARY STATISTICS:\n")
        f.write("-" * 40 + "\n")
        summary = history['summary']
        f.write(f"Total Steps: {summary['total_steps']}\n")
        f.write(f"Total States Saved: {summary['total_states_saved']}\n")
        f.write(f"Total Predictions Saved: {summary['total_predictions_saved']}\n")
        f.write(f"Finalized Tokens Count: {summary['finalized_tokens_count']}\n")
        f.write(f"Generation Completed: {summary['generation_completed']}\n\n")
        
        # Detailed Step-by-Step Analysis
        f.write("DETAILED STEP-BY-STEP ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        
        for i, (state, step_info, prediction, remask_decision) in enumerate(zip(
            history['states'][:-1],  # Exclude final state
            history['step_info'],
            history['predictions'],
            history['remasking_decisions']
        )):
            f.write(f"\nStep {step_info['global_step']} (Block {step_info['block']}, Block Step {step_info['block_step']}):\n")
            f.write(f"  Block Range: {step_info['block_start']}-{step_info['block_end']-1}\n")
            
            # Show generated tokens so far
            current_tokens = state[0].cpu().tolist()
            prompt_len = history['prompt_length']
            generated_tokens = current_tokens[prompt_len:]
            
            if any(t != history['mask_id'] for t in generated_tokens):
                # Find the last non-mask token
                last_finalized = max((i for i, t in enumerate(generated_tokens) if t != history['mask_id']), default=-1)
                if last_finalized >= 0:
                    finalized_so_far = generated_tokens[:last_finalized + 1]
                    finalized_text = tokenizer.decode(finalized_so_far, skip_special_tokens=True)
                    f.write(f"  Generated so far: {repr(finalized_text)}\n")
            
            f.write("\n")  # Newline between generated so far and model prediction
            
            # Show prediction for this step
            pred_tokens = prediction[0].cpu().tolist()
            pred_generated = pred_tokens[prompt_len:]
            if any(t != history['mask_id'] for t in pred_generated):
                pred_text = tokenizer.decode(pred_generated, skip_special_tokens=True)
                f.write(f"  Model prediction: {repr(pred_text)}\n")
            
            # Show remasking decision
            f.write(f"  Tokens finalized: {remask_decision['tokens_finalized']}\n")
            if remask_decision['finalized_positions']:
                f.write(f"  Finalized positions: {remask_decision['finalized_positions']}\n")
                # Show what tokens were finalized
                finalized_tokens_this_step = [pred_tokens[pos] for pos in remask_decision['finalized_positions']]
                finalized_text_this_step = tokenizer.decode(finalized_tokens_this_step, skip_special_tokens=True)
                f.write(f"  Finalized text: {repr(finalized_text_this_step)}\n")
        
        # Final State
        f.write(f"\nFINAL STATE:\n")
        f.write("-" * 40 + "\n")
        final_state = history['states'][-1][0].cpu().tolist()
        final_text = tokenizer.decode(final_state, skip_special_tokens=True)
        f.write(f"Complete generated text:\n{repr(final_text)}\n")
        
        # Token-level analysis of final state
        f.write(f"\nTOKEN-LEVEL ANALYSIS OF FINAL STATE:\n")
        f.write("-" * 40 + "\n")
        prompt_len = history['prompt_length']
        f.write(f"Prompt tokens (0-{prompt_len-1}): {final_state[:prompt_len]}\n")
        f.write(f"Generated tokens ({prompt_len}-{len(final_state)-1}): {final_state[prompt_len:]}\n")
        
        # Show token IDs and their decoded text
        f.write(f"\nDetailed token breakdown:\n")
        for i, token_id in enumerate(final_state):
            token_text = tokenizer.decode([token_id], skip_special_tokens=True)
            if i < prompt_len:
                f.write(f"  {i:3d}: {token_id:6d} (prompt) -> {repr(token_text)}\n")
            else:
                f.write(f"  {i:3d}: {token_id:6d} (gen)   -> {repr(token_text)}\n")
    
    print(f"Generation history saved to: {output_file}")
    return output_file


def generate_text(prompt, gen_length=128, steps=128, block_length=32, temperature=1.0, 
                  cfg_scale=0.0, remasking='low_confidence', save_history=False, use_chat_template=True, enable_cot=False, seed=None):
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
            cfg_scale=cfg_scale, remasking=remasking, save_history=True, seed=seed
        )
        generated_text = _tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        
        # Save generation history to text file for human inspection
        history_file = save_generation_history_to_file(history, _tokenizer)
        
        # Return the history dictionary directly for immediate analysis
        return generated_text, history, _tokenizer, history_file
    else:
        generated_ids = mmada_generate(
            _model, input_ids, steps=steps, gen_length=gen_length,
            block_length=block_length, temperature=temperature,
            cfg_scale=cfg_scale, remasking=remasking, save_history=False, seed=seed
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
                
                if save_history:
                    # result is now (text, history, tokenizer, history_file)
                    batch_results.append(result)
                else:
                    # result is just the text
                    batch_results.append(result)
            
            results.extend(batch_results)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    finally:
        if unload_after:
            _unload_model()
    
    return results
