"""
Custom Inspect AI Model API for MMaDA-8B-MixCoT
"""
import torch
from typing import Any, List
from inspect_ai.model import ModelAPI, GenerateConfig, ChatMessage, ModelOutput, ChatMessageUser, ChatMessageAssistant
from inspect_ai.tool import ToolInfo, ToolChoice
from transformers import AutoTokenizer
import sys
import os

# Add the parent directory to the path to import our MMaDA inference module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from mmada_inference import generate_text


class MMadaModelAPI(ModelAPI):
    """Custom ModelAPI implementation for MMaDA-8B-MixCoT"""
    
    # Singleton pattern to avoid multiple model loads
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] = [],
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any
    ) -> None:
        if self._initialized:
            return
        super().__init__(model_name, base_url, api_key, api_key_vars, config)
        
        # Store any custom model arguments
        self.model_args = model_args
        
        # Set default generation parameters
        self.default_gen_length = getattr(config, 'max_tokens', 256) or 256
        self.default_temperature = getattr(config, 'temperature', 1.0) or 1.0
        self.default_steps = getattr(config, 'steps', 128) or 128
        self.default_block_length = getattr(config, 'block_length', 32) or 32
        self.default_cfg_scale = getattr(config, 'cfg_scale', 0.0) or 0.0
        self.default_remasking = getattr(config, 'remasking', 'low_confidence') or 'low_confidence'
        self.default_use_chat_template = getattr(config, 'use_chat_template', True)
        self.default_enable_cot = getattr(config, 'enable_cot', False)
        
        print(f"MMaDA Model API initialized with model: {model_name}")
        print(f"Default parameters: gen_length={self.default_gen_length}, temperature={self.default_temperature}")
        self._initialized = True
    
    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        """Generate a response using the MMaDA model"""
        
        # Convert Inspect AI messages to a single prompt
        prompt = self._convert_messages_to_prompt(input)
        
        # Get generation parameters from config, with fallbacks to defaults
        gen_length = getattr(config, 'max_tokens', self.default_gen_length) or self.default_gen_length
        temperature = getattr(config, 'temperature', self.default_temperature) or self.default_temperature
        steps = getattr(config, 'steps', self.default_steps) or self.default_steps
        block_length = getattr(config, 'block_length', self.default_block_length) or self.default_block_length
        cfg_scale = getattr(config, 'cfg_scale', self.default_cfg_scale) or self.default_cfg_scale
        remasking = getattr(config, 'remasking', self.default_remasking) or self.default_remasking
        use_chat_template = getattr(config, 'use_chat_template', self.default_use_chat_template)
        enable_cot = getattr(config, 'enable_cot', self.default_enable_cot)
        
        # Generate response using MMaDA
        response_text = await self._generate_response(
            prompt, 
            gen_length=gen_length,
            temperature=temperature,
            steps=steps,
            block_length=block_length,
            cfg_scale=cfg_scale,
            remasking=remasking,
            use_chat_template=use_chat_template,
            enable_cot=enable_cot
        )
        
        # Create ModelOutput
        return ModelOutput.from_content(
            model=self.model_name,
            content=response_text
        )
    
    def _convert_messages_to_prompt(self, messages: list[ChatMessage]) -> str:
        """Convert Inspect AI ChatMessage list to MMaDA prompt format"""
        
        # For MMaDA, we'll concatenate all messages into a single prompt
        # This is simpler than the chat template approach for evaluation
        prompt_parts = []
        
        for message in messages:
            if isinstance(message, ChatMessageUser):
                content = message.content
                if isinstance(content, list):
                    # Handle multimodal content - extract text
                    text_parts = [item.text for item in content if hasattr(item, 'text')]
                    content = " ".join(text_parts)
                
                # Don't add "User: " prefix - just use the content directly
                prompt_parts.append(content.strip())
            elif isinstance(message, ChatMessageAssistant):
                content = message.content or ""
                # Don't add "Assistant: " prefix - just use the content directly
                prompt_parts.append(content.strip())
        
        # Join all parts with newlines
        return "\n".join(prompt_parts)
    
    def _generate_response_sync(
        self, 
        prompt: str, 
        gen_length: int = 256,
        temperature: float = 1.0,
        steps: int = 128,
        block_length: int = 32,
        cfg_scale: float = 0.0,
        remasking: str = 'low_confidence',
        use_chat_template: bool = True,
        enable_cot: bool = False
    ) -> str:
        """Generate response using the MMaDA model (synchronous)"""
        
        try:
            # Use our existing generate_text function
            response = generate_text(
                prompt=prompt,
                gen_length=gen_length,
                steps=steps,
                block_length=block_length,
                temperature=temperature,
                cfg_scale=cfg_scale,
                remasking=remasking,
                save_history=False,
                use_chat_template=use_chat_template,
                enable_cot=enable_cot
            )
            
            return response.strip()
            
        except Exception as e:
            print(f"Error in MMaDA generation: {e}")
            return f"Error: {str(e)}"
    
    async def _generate_response(
        self, 
        prompt: str, 
        gen_length: int = 256,
        temperature: float = 1.0,
        steps: int = 128,
        block_length: int = 32,
        cfg_scale: float = 0.0,
        remasking: str = 'low_confidence',
        use_chat_template: bool = True,
        enable_cot: bool = False
    ) -> str:
        """Generate response using the MMaDA model (async wrapper)"""
        # Run the synchronous generation in a thread pool to avoid blocking
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self._generate_response_sync, 
            prompt, 
            gen_length, 
            temperature, 
            steps, 
            block_length, 
            cfg_scale, 
            remasking, 
            use_chat_template, 
            enable_cot
        )


# Model provider registration
from inspect_ai.model import modelapi

class MMadaProvider:
    """MMaDA model provider that handles any model ID"""
    
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] = [],
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any
    ):
        # Create the actual model API instance
        self.model_api = MMadaModelAPI(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            api_key_vars=api_key_vars,
            config=config,
            **model_args
        )
    
    def __getattr__(self, name):
        # Delegate all method calls to the underlying model API
        return getattr(self.model_api, name)

@modelapi(name="mmada")
def mmada_provider():
    """Create MMaDA provider"""
    return MMadaProvider

print("MMaDA provider registered successfully!")
