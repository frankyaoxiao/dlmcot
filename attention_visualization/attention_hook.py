"""
Attention Hook for MMaDA Models

This module provides the core functionality to hook into attention mechanisms
and capture attention weights during model inference.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import logging
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

class AttentionHook:
    """
    Hooks into attention mechanisms of MMaDA models to capture attention weights.
    
    This class registers forward hooks on attention modules to capture
    attention weights during text generation, enabling visualization and analysis
    of attention patterns.
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize the attention hook.
        
        Args:
            model: The MMaDA model to hook into
        """
        self.model = model
        self.attention_maps = defaultdict(list)
        self.hooks = []
        self.is_active = False
        self.current_step = 0
        self.current_block = 0
        
        # Track which modules we've hooked into
        self.hooked_modules = set()
        
        # Track conversion statistics
        self.conversion_stats = {
            'bfloat16_conversions': 0,
            'float16_conversions': 0,
            'conversion_errors': 0,
            'storage_errors': 0
        }
        
        logger.info("AttentionHook initialized")
    
    def _safe_tensor_conversion(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Safely convert tensor to CPU float32 for numpy compatibility.
        
        Args:
            tensor: Input tensor that may be BFloat16, Float16, or other dtype
            
        Returns:
            Tensor converted to CPU float32
        """
        try:
            if tensor.dtype == torch.bfloat16:
                self.conversion_stats['bfloat16_conversions'] += 1
                return tensor.detach().cpu().float()
            elif tensor.dtype == torch.float16:
                self.conversion_stats['float16_conversions'] += 1
                return tensor.detach().cpu().float()
            else:
                return tensor.detach().cpu()
        except Exception as e:
            self.conversion_stats['conversion_errors'] += 1
            logger.warning(f"Failed to convert tensor from {tensor.dtype}: {e}")
            # Return original tensor as fallback
            return tensor
    
    def _safe_tensor_stats(self, tensor: torch.Tensor) -> dict:
        """
        Safely compute tensor statistics with dtype conversion.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Dictionary with tensor statistics
        """
        try:
            tensor_converted = self._safe_tensor_conversion(tensor)
            
            return {
                'mean': float(tensor_converted.mean()),
                'std': float(tensor_converted.std()),
                'min': float(tensor_converted.min()),
                'max': float(tensor_converted.max()),
                'dtype': str(tensor.dtype),
                'converted_dtype': str(tensor_converted.dtype)
            }
        except Exception as e:
            self.conversion_stats['conversion_errors'] += 1
            return {
                'mean': None,
                'std': None,
                'min': None,
                'max': None,
                'dtype': str(tensor.dtype),
                'error': str(e)
            }
    
    def register_hooks(self) -> None:
        """
        Register forward hooks on all attention-related modules in the model.
        
        This method traverses the model and hooks into:
        - Attention projection layers (q, k, v projections)
        - Scaled dot product attention computation
        - Attention output layers
        """
        logger.info("Registering attention hooks...")
        
        # Clear any existing hooks
        self.remove_hooks()
        
        # Hook into attention mechanisms
        self._hook_attention_modules()
        
        # Hook into the scaled dot product attention method
        self._hook_scaled_dot_product_attention()
        
        self.is_active = True
        logger.info(f"Registered {len(self.hooks)} attention hooks")
    
    def _hook_attention_modules(self) -> None:
        """Hook into attention projection and computation modules."""
        
        for name, module in self.model.named_modules():
            # Hook into attention projection layers
            if any(keyword in name.lower() for keyword in ['att_proj', 'q_proj', 'k_proj', 'v_proj']):
                if module not in self.hooked_modules:
                    hook = module.register_forward_hook(self._attention_projection_hook)
                    self.hooks.append(hook)
                    self.hooked_modules.add(module)
                    logger.debug(f"Hooked into attention projection: {name}")
            
            # Hook into attention computation layers
            elif 'attention' in name.lower() and hasattr(module, 'forward'):
                if module not in self.hooked_modules:
                    hook = module.register_forward_hook(self._attention_computation_hook)
                    self.hooks.append(hook)
                    self.hooked_modules.add(module)
                    logger.debug(f"Hooked into attention computation: {name}")
    
    def _hook_scaled_dot_product_attention(self) -> None:
        """Hook into the scaled dot product attention method."""
        
        # Find the LLaDABlock classes that contain the attention method
        for name, module in self.model.named_modules():
            if hasattr(module, '_scaled_dot_product_attention'):
                # Store the original method
                original_method = module._scaled_dot_product_attention
                
                # Create a wrapper that captures attention weights
                def make_attention_wrapper(original_method, module_name):
                    def wrapper(*args, **kwargs):
                        # Call the original method
                        result = original_method(*args, **kwargs)
                        
                        # Extract attention weights if available
                        if len(args) >= 3:
                            q, k, v = args[:3]
                            attn_mask = kwargs.get('attn_mask', None)
                            
                            # Compute attention weights manually
                            try:
                                attn_weights = self._compute_attention_weights(q, k, attn_mask)
                                self._store_attention_weights(module_name, attn_weights)
                            except Exception as e:
                                logger.warning(f"Failed to capture attention weights from {module_name}: {e}")
                        
                        return result
                    
                    return wrapper
                
                # Replace the method temporarily
                wrapper = make_attention_wrapper(original_method, name)
                setattr(module, '_scaled_dot_product_attention', wrapper)
                
                # Store the original method for restoration
                self.hooks.append((module, '_scaled_dot_product_attention', original_method))
                logger.debug(f"Hooked into scaled dot product attention: {name}")
    
    def _attention_projection_hook(self, module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor) -> None:
        """Hook function for attention projection layers."""
        try:
            # Store Q, K, V projections for analysis
            layer_name = self._get_module_name(module)
            if 'att_proj' in layer_name:
                # This is the fused QKV projection
                self._store_qkv_projections(layer_name, input[0], output)
            elif any(keyword in layer_name for keyword in ['q_proj', 'k_proj', 'v_proj']):
                # Individual Q, K, or V projection
                self._store_individual_projection(layer_name, input[0], output)
        except Exception as e:
            logger.warning(f"Error in attention projection hook: {e}")
    
    def _attention_computation_hook(self, module: nn.Module, input: Tuple[torch.Tensor], output: Tuple[torch.Tensor, ...]) -> None:
        """Hook function for attention computation layers."""
        try:
            layer_name = self._get_module_name(module)
            
            # Extract attention output and cache
            if isinstance(output, tuple):
                attn_output = output[0]
                cache = output[1] if len(output) > 1 else None
            else:
                attn_output = output
                cache = None
            
            # Store attention output for analysis
            self._store_attention_output(layer_name, attn_output, cache)
            
        except Exception as e:
            logger.warning(f"Error in attention computation hook: {e}")
    
    def _compute_attention_weights(self, q: torch.Tensor, k: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute attention weights from query and key tensors.
        
        Args:
            q: Query tensor of shape (batch_size, n_heads, seq_len, head_dim)
            k: Key tensor of shape (batch_size, n_heads, seq_len, head_dim)
            attn_mask: Optional attention mask
            
        Returns:
            Attention weights tensor of shape (batch_size, n_heads, seq_len, seq_len)
        """
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1))
        
        # Scale by head dimension
        head_dim = q.size(-1)
        scores = scores / (head_dim ** 0.5)
        
        # Apply attention mask if provided
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        
        return attn_weights
    
    def _store_attention_weights(self, module_name: str, attn_weights: torch.Tensor) -> None:
        """Store attention weights for the current step."""
        try:
            # Use safe tensor conversion utility
            attn_weights_converted = self._safe_tensor_conversion(attn_weights)
            
            # Convert to numpy for storage (more memory efficient)
            attn_weights_np = attn_weights_converted.numpy()
            
            # Store with metadata
            attention_data = {
                'step': self.current_step,
                'block': self.current_block,
                'attention_weights': attn_weights_np,
                'shape': attn_weights_np.shape,
                'dtype': str(attn_weights.dtype),
                'converted_dtype': str(attn_weights_converted.dtype),
                'timestamp': torch.cuda.Event() if torch.cuda.is_available() else None
            }
            
            self.attention_maps[module_name].append(attention_data)
            
            logger.debug(f"Stored attention weights for {module_name} at step {self.current_step} (converted from {attn_weights.dtype} to {attn_weights_converted.dtype})")
            
        except Exception as e:
            self.conversion_stats['storage_errors'] += 1
            logger.warning(f"Failed to store attention weights: {e}")
            # Store minimal information even if conversion fails
            try:
                attention_data = {
                    'step': self.current_step,
                    'block': self.current_block,
                    'attention_weights': None,
                    'shape': attn_weights.shape,
                    'dtype': str(attn_weights.dtype),
                    'error': str(e),
                    'timestamp': torch.cuda.Event() if torch.cuda.is_available() else None
                }
                self.attention_maps[module_name].append(attention_data)
                logger.info(f"Stored error information for {module_name} at step {self.current_step}")
            except Exception as e2:
                logger.error(f"Failed to store even error information: {e2}")
    
    def _store_qkv_projections(self, layer_name: str, input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> None:
        """Store QKV projections for analysis."""
        try:
            # Store input and output shapes for analysis
            projection_data = {
                'step': self.current_step,
                'block': self.current_block,
                'input_shape': input_tensor.shape,
                'output_shape': output_tensor.shape,
                'input_stats': self._safe_tensor_stats(input_tensor),
                'output_stats': self._safe_tensor_stats(output_tensor)
            }
            
            self.attention_maps[f"{layer_name}_projections"].append(projection_data)
            
        except Exception as e:
            self.conversion_stats['storage_errors'] += 1
            logger.warning(f"Failed to store QKV projections: {e}")
            # Store minimal information even if statistics fail
            try:
                projection_data = {
                    'step': self.current_step,
                    'block': self.current_block,
                    'input_shape': input_tensor.shape,
                    'output_shape': output_tensor.shape,
                    'input_stats': {'error': str(e)},
                    'output_stats': {'error': str(e)}
                }
                self.attention_maps[f"{layer_name}_projections"].append(projection_data)
            except Exception as e2:
                logger.error(f"Failed to store even minimal projection data: {e2}")
    
    def _store_individual_projection(self, layer_name: str, input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> None:
        """Store individual Q, K, or V projection data."""
        try:
            projection_data = {
                'step': self.current_step,
                'block': self.current_block,
                'input_shape': input_tensor.shape,
                'output_shape': output_tensor.shape,
                'input_stats': self._safe_tensor_stats(input_tensor),
                'output_stats': self._safe_tensor_stats(output_tensor)
            }
            
            self.attention_maps[f"{layer_name}_projection"].append(projection_data)
            
        except Exception as e:
            self.conversion_stats['storage_errors'] += 1
            logger.warning(f"Failed to store individual projection: {e}")
            # Store minimal information even if statistics fail
            try:
                projection_data = {
                    'step': self.current_step,
                    'block': self.current_block,
                    'input_shape': input_tensor.shape,
                    'output_shape': output_tensor.shape,
                    'input_stats': {'error': str(e)},
                    'output_stats': {'error': str(e)}
                }
                self.attention_maps[f"{layer_name}_projection"].append(projection_data)
            except Exception as e2:
                logger.error(f"Failed to store even minimal individual projection data: {e2}")
    
    def _store_attention_output(self, layer_name: str, attn_output: torch.Tensor, cache: Optional[Tuple] = None) -> None:
        """Store attention output for analysis."""
        try:
            output_data = {
                'step': self.current_step,
                'block': self.current_block,
                'output_shape': attn_output.shape,
                'output_stats': self._safe_tensor_stats(attn_output),
                'has_cache': cache is not None
            }
            
            self.attention_maps[f"{layer_name}_output"].append(output_data)
            
        except Exception as e:
            self.conversion_stats['storage_errors'] += 1
            logger.warning(f"Failed to store attention output: {e}")
            # Store minimal information even if statistics fail
            try:
                output_data = {
                    'step': self.current_step,
                    'block': self.current_block,
                    'output_shape': attn_output.shape,
                    'output_stats': {'error': str(e)},
                    'max': None,
                    'has_cache': cache is not None
                }
                self.attention_maps[f"{layer_name}_output"].append(output_data)
            except Exception as e2:
                logger.error(f"Failed to store even minimal attention output data: {e2}")
    
    def _get_module_name(self, module: nn.Module) -> str:
        """Get a unique name for a module."""
        for name, mod in self.model.named_modules():
            if mod is module:
                return name
        return f"unknown_module_{id(module)}"
    
    def set_generation_context(self, step: int, block: int) -> None:
        """
        Set the current generation context for attention tracking.
        
        Args:
            step: Current diffusion step
            block: Current generation block
        """
        self.current_step = step
        self.current_block = block
        logger.debug(f"Set generation context: step={step}, block={block}")
    
    def get_attention_maps(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all captured attention maps.
        
        Returns:
            Dictionary mapping module names to lists of attention data
        """
        return dict(self.attention_maps)
    
    def get_attention_maps_for_step(self, step: int) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get attention maps for a specific generation step.
        
        Args:
            step: Generation step to filter by
            
        Returns:
            Dictionary mapping module names to attention data for the specified step
        """
        filtered_maps = {}
        for module_name, attention_data_list in self.attention_maps.items():
            filtered_data = [data for data in attention_data_list if data['step'] == step]
            if filtered_data:
                filtered_maps[module_name] = filtered_data
        
        return filtered_maps
    
    def get_attention_maps_for_block(self, block: int) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get attention maps for a specific generation block.
        
        Args:
            block: Generation block to filter by
            
        Returns:
            Dictionary mapping module names to attention data for the specified block
        """
        filtered_maps = {}
        for module_name, attention_data_list in self.attention_maps.items():
            filtered_data = [data for data in attention_data_list if data['block'] == block]
            if filtered_data:
                filtered_maps[module_name] = filtered_data
        
        return filtered_maps
    
    def clear_attention_maps(self) -> None:
        """Clear all stored attention maps."""
        self.attention_maps.clear()
        self.current_step = 0
        self.current_block = 0
        logger.info("Cleared all attention maps")
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks and restore original methods."""
        logger.info("Removing attention hooks...")
        
        # Remove forward hooks
        for hook in self.hooks:
            if hasattr(hook, 'remove'):
                hook.remove()
            elif isinstance(hook, tuple):
                # This is a method replacement hook
                module, method_name, original_method = hook
                setattr(module, method_name, original_method)
        
        self.hooks.clear()
        self.hooked_modules.clear()
        self.is_active = False
        
        logger.info("Removed all attention hooks")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics about captured attention data.
        
        Returns:
            Dictionary containing summary statistics
        """
        total_attention_maps = sum(len(data_list) for data_list in self.attention_maps.values())
        
        # Count attention weights captures
        attention_weights_count = 0
        total_attention_parameters = 0
        
        for module_name, data_list in self.attention_maps.items():
            for data in data_list:
                if 'attention_weights' in data and data['attention_weights'] is not None:
                    attention_weights_count += 1
                    if hasattr(data['attention_weights'], 'shape'):
                        total_attention_parameters += np.prod(data['attention_weights'].shape)
        
        summary = {
            'total_modules_hooked': len(self.hooked_modules),
            'total_attention_maps': total_attention_maps,
            'attention_weights_captured': attention_weights_count,
            'total_attention_parameters': total_attention_parameters,
            'is_active': self.is_active,
            'current_step': self.current_step,
            'current_block': self.current_block,
            'conversion_statistics': self.conversion_stats.copy()
        }
        
        return summary
    
    def get_conversion_stats(self) -> Dict[str, int]:
        """
        Get conversion statistics for debugging and monitoring.
        
        Returns:
            Dictionary containing conversion statistics
        """
        return self.conversion_stats.copy()
    
    def print_conversion_summary(self) -> None:
        """Print a summary of conversion statistics."""
        print("🔍 Attention Hook Conversion Summary:")
        print(f"  BFloat16 conversions: {self.conversion_stats['bfloat16_conversions']}")
        print(f"  Conversion errors: {self.conversion_stats['conversion_errors']}")
        print(f"  Storage errors: {self.conversion_stats['storage_errors']}")
    
    def __del__(self):
        """Cleanup when the hook is destroyed."""
        try:
            self.remove_hooks()
        except:
            pass
