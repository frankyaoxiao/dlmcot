"""
Attention Analyzer for MMaDA Models

This module provides high-level analysis and insights from captured attention data,
including attention pattern analysis, token influence analysis, and statistical summaries.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from collections import defaultdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class AttentionAnalyzer:
    """
    Analyzes attention patterns and provides insights from captured attention data.
    
    This class provides various analysis methods including:
    - Attention pattern analysis
    - Token influence analysis
    - Statistical summaries
    - Attention evolution analysis
    """
    
    def __init__(self, attention_maps: Dict[str, List[Dict[str, Any]]]):
        """
        Initialize the attention analyzer.
        
        Args:
            attention_maps: Dictionary of attention maps from AttentionHook
        """
        self.attention_maps = attention_maps
        self.analysis_cache = {}
        
        logger.info("AttentionAnalyzer initialized")
    
    def analyze_attention_patterns(self) -> Dict[str, Any]:
        """
        Analyze overall attention patterns across all layers and steps.
        
        Returns:
            Dictionary containing attention pattern analysis
        """
        if 'attention_patterns' in self.analysis_cache:
            return self.analysis_cache['attention_patterns']
        
        logger.info("Analyzing attention patterns...")
        
        analysis = {
            'total_layers': 0,
            'total_steps': 0,
            'attention_density': {},
            'attention_entropy': {},
            'attention_focus': {},
            'cross_attention_patterns': {},
            'layer_similarity': {}
        }
        
        # Collect all unique layers and steps
        layers = set()
        steps = set()
        
        for module_name, data_list in self.attention_maps.items():
            # Consider all modules as layers for now
            layers.add(module_name)
            for data in data_list:
                if 'step' in data:
                    steps.add(data['step'])
        
        analysis['total_layers'] = len(layers)
        analysis['total_steps'] = len(steps)
        
        # Analyze attention density and entropy for each layer
        for layer_name in layers:
            layer_data = self.attention_maps[layer_name]
            
            if not layer_data:
                continue
            
            # Analyze attention weights
            attention_weights_list = []
            for data in layer_data:
                if 'attention_weights' in data:
                    attention_weights_list.append(data['attention_weights'])
            
            if not attention_weights_list:
                continue
            
            # Compute attention density (how focused vs. diffuse attention is)
            density_scores = []
            entropy_scores = []
            focus_scores = []
            
            for attn_weights in attention_weights_list:
                # Handle different tensor shapes and convert to numpy
                if hasattr(attn_weights, 'detach'):
                    attn_weights = attn_weights.detach().cpu().numpy()
                
                if attn_weights.ndim == 4:
                    attn_weights = attn_weights[0].mean(axis=0)  # Average across heads
                elif attn_weights.ndim == 3:
                    attn_weights = attn_weights.mean(axis=0)  # Average across heads
                
                # Attention density: how concentrated attention is
                max_attention = float(attn_weights.max())
                mean_attention = float(attn_weights.mean())
                density = max_attention / (mean_attention + 1e-8)
                density_scores.append(density)
                
                # Attention entropy: how uniform vs. focused attention is
                # Normalize attention weights to probability distribution
                attn_probs = attn_weights / (attn_weights.sum() + 1e-8)
                entropy = -float(np.sum(attn_probs * np.log(attn_probs + 1e-8)))
                entropy_scores.append(entropy)
                
                # Attention focus: how much attention is on diagonal vs. off-diagonal
                seq_len = attn_weights.shape[0]
                diagonal_attention = float(np.trace(attn_weights))
                total_attention = float(attn_weights.sum())
                focus = diagonal_attention / (total_attention + 1e-8)
                focus_scores.append(focus)
            
            # Store statistics
            analysis['attention_density'][layer_name] = {
                'mean': np.mean(density_scores),
                'std': np.std(density_scores),
                'min': np.min(density_scores),
                'max': np.max(density_scores)
            }
            
            analysis['attention_entropy'][layer_name] = {
                'mean': np.mean(entropy_scores),
                'std': np.std(entropy_scores),
                'min': np.min(entropy_scores),
                'max': np.max(entropy_scores)
            }
            
            analysis['attention_focus'][layer_name] = {
                'mean': np.mean(focus_scores),
                'std': np.std(focus_scores),
                'min': np.min(focus_scores),
                'max': np.max(focus_scores)
            }
        
        # Analyze cross-attention patterns between prompt and generated tokens
        analysis['cross_attention_patterns'] = self._analyze_cross_attention_patterns()
        
        # Analyze layer similarity
        analysis['layer_similarity'] = self._analyze_layer_similarity()
        
        # Cache the analysis
        self.analysis_cache['attention_patterns'] = analysis
        
        logger.info("Attention pattern analysis completed")
        return analysis
    
    def analyze_token_influence(
        self, 
        prompt_length: int,
        max_tokens: int = 100
    ) -> Dict[str, Any]:
        """
        Analyze how much influence each token has on the generation process.
        
        Args:
            prompt_length: Length of the prompt (to distinguish prompt vs. generated tokens)
            max_tokens: Maximum number of tokens to analyze
            
        Returns:
            Dictionary containing token influence analysis
        """
        cache_key = f"token_influence_{prompt_length}_{max_tokens}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        logger.info("Analyzing token influence...")
        
        analysis = {
            'prompt_tokens': {},
            'generated_tokens': {},
            'cross_token_influence': {},
            'attention_flow': {}
        }
        
        # Collect attention data for all layers and steps
        all_attention_weights = []
        all_steps = []
        all_layers = []
        
        for module_name, data_list in self.attention_maps.items():
            if 'attention' in module_name.lower():
                for data in data_list:
                    if 'attention_weights' in data:
                        all_attention_weights.append(data['attention_weights'])
                        all_steps.append(data.get('step', 0))
                        all_layers.append(module_name)
        
        if not all_attention_weights:
            logger.warning("No attention weights found for token influence analysis")
            return analysis
        
        # Analyze each attention weight matrix
        for i, attn_weights in enumerate(all_attention_weights):
            step = all_steps[i]
            layer = all_layers[i]
            
            # Handle different tensor shapes
            if attn_weights.ndim == 4:
                attn_weights = attn_weights[0].mean(axis=0)  # Average across heads
            elif attn_weights.ndim == 3:
                attn_weights = attn_weights.mean(axis=0)  # Average across heads
            
            # Limit tokens for analysis
            if attn_weights.shape[0] > max_tokens:
                attn_weights = attn_weights[:max_tokens, :max_tokens]
            
            # Analyze token influence
            self._analyze_single_attention_matrix(
                attn_weights, step, layer, prompt_length, analysis
            )
        
        # Aggregate results across steps and layers
        analysis = self._aggregate_token_influence(analysis)
        
        # Cache the analysis
        self.analysis_cache[cache_key] = analysis
        
        logger.info("Token influence analysis completed")
        return analysis
    
    def _analyze_single_attention_matrix(
        self, 
        attn_weights: np.ndarray, 
        step: int, 
        layer: str, 
        prompt_length: int, 
        analysis: Dict[str, Any]
    ) -> None:
        """Analyze a single attention matrix for token influence."""
        
        seq_len = attn_weights.shape[0]
        
        # Analyze how much each token influences others (outgoing attention)
        for token_idx in range(min(seq_len, prompt_length + 50)):  # Limit analysis
            # How much this token influences others
            outgoing_influence = attn_weights[:, token_idx].sum()
            
            # How much this token is influenced by others
            incoming_influence = attn_weights[token_idx, :].sum()
            
            # Store results
            if token_idx < prompt_length:
                # This is a prompt token
                if token_idx not in analysis['prompt_tokens']:
                    analysis['prompt_tokens'][token_idx] = {
                        'outgoing_influence': [],
                        'incoming_influence': [],
                        'steps': [],
                        'layers': []
                    }
                
                analysis['prompt_tokens'][token_idx]['outgoing_influence'].append(outgoing_influence)
                analysis['prompt_tokens'][token_idx]['incoming_influence'].append(incoming_influence)
                analysis['prompt_tokens'][token_idx]['steps'].append(step)
                analysis['prompt_tokens'][token_idx]['layers'].append(layer)
            else:
                # This is a generated token
                gen_idx = token_idx - prompt_length
                if gen_idx not in analysis['generated_tokens']:
                    analysis['generated_tokens'][gen_idx] = {
                        'outgoing_influence': [],
                        'incoming_influence': [],
                        'steps': [],
                        'layers': []
                    }
                
                analysis['generated_tokens'][gen_idx]['outgoing_influence'].append(outgoing_influence)
                analysis['generated_tokens'][gen_idx]['incoming_influence'].append(incoming_influence)
                analysis['generated_tokens'][gen_idx]['steps'].append(step)
                analysis['generated_tokens'][gen_idx]['layers'].append(layer)
        
        # Analyze cross-attention between prompt and generated tokens
        if prompt_length < seq_len:
            prompt_to_gen = attn_weights[prompt_length:, :prompt_length].sum()
            gen_to_prompt = attn_weights[:prompt_length, prompt_length:].sum()
            
            if step not in analysis['cross_token_influence']:
                analysis['cross_token_influence'][step] = {}
            
            analysis['cross_token_influence'][step][layer] = {
                'prompt_to_generated': prompt_to_gen,
                'generated_to_prompt': gen_to_prompt,
                'ratio': prompt_to_gen / (gen_to_prompt + 1e-8)
            }
    
    def _aggregate_token_influence(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate token influence results across steps and layers."""
        
        # Aggregate prompt token influence
        for token_idx, token_data in analysis['prompt_tokens'].items():
            if token_data['outgoing_influence']:
                token_data['avg_outgoing'] = np.mean(token_data['outgoing_influence'])
                token_data['avg_incoming'] = np.mean(token_data['incoming_influence'])
                token_data['total_steps'] = len(token_data['steps'])
                token_data['total_layers'] = len(set(token_data['layers']))
        
        # Aggregate generated token influence
        for token_idx, token_data in analysis['generated_tokens'].items():
            if token_data['outgoing_influence']:
                token_data['avg_outgoing'] = np.mean(token_data['outgoing_influence'])
                token_data['avg_incoming'] = np.mean(token_data['incoming_influence'])
                token_data['total_steps'] = len(token_data['steps'])
                token_data['total_layers'] = len(set(token_data['layers']))
        
        # Aggregate cross-token influence
        if analysis['cross_token_influence']:
            all_prompt_to_gen = []
            all_gen_to_prompt = []
            all_ratios = []
            
            for step_data in analysis['cross_token_influence'].values():
                for layer_data in step_data.values():
                    all_prompt_to_gen.append(layer_data['prompt_to_generated'])
                    all_gen_to_prompt.append(layer_data['generated_to_prompt'])
                    all_ratios.append(layer_data['ratio'])
            
            analysis['cross_token_influence_summary'] = {
                'avg_prompt_to_generated': np.mean(all_prompt_to_gen),
                'avg_generated_to_prompt': np.mean(all_gen_to_prompt),
                'avg_ratio': np.mean(all_ratios),
                'total_observations': len(all_prompt_to_gen)
            }
        
        return analysis
    
    def _analyze_cross_attention_patterns(self) -> Dict[str, Any]:
        """Analyze cross-attention patterns between different parts of the sequence."""
        
        analysis = {
            'prompt_self_attention': {},
            'generated_self_attention': {},
            'cross_sequence_attention': {}
        }
        
        # This is a placeholder for cross-attention analysis
        # Implementation would depend on the specific attention patterns in MMaDA
        
        return analysis
    
    def _analyze_layer_similarity(self) -> Dict[str, Any]:
        """Analyze similarity between attention patterns across layers."""
        
        analysis = {
            'layer_correlations': {},
            'attention_pattern_clustering': {}
        }
        
        # This is a placeholder for layer similarity analysis
        # Implementation would compute correlations between attention patterns
        
        return analysis
    
    def get_attention_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the attention data.
        
        Returns:
            Dictionary containing attention statistics
        """
        if 'statistics' in self.analysis_cache:
            return self.analysis_cache['statistics']
        
        logger.info("Computing attention statistics...")
        
        stats = {
            'total_attention_maps': 0,
            'attention_weights_count': 0,
            'total_parameters': 0,
            'steps_coverage': set(),
            'layers_coverage': set(),
            'attention_shapes': [],
            'memory_usage_estimate': 0
        }
        
        for module_name, data_list in self.attention_maps.items():
            stats['total_attention_maps'] += len(data_list)
            stats['layers_coverage'].add(module_name)
            
            for data in data_list:
                if 'step' in data:
                    stats['steps_coverage'].add(data['step'])
                
                if 'attention_weights' in data:
                    stats['attention_weights_count'] += 1
                    stats['attention_shapes'].append(data['shape'])
                    stats['total_parameters'] += np.prod(data['shape'])
        
        # Convert sets to lists for JSON serialization
        stats['steps_coverage'] = sorted(list(stats['steps_coverage']))
        stats['layers_coverage'] = list(stats['layers_coverage'])
        
        # Estimate memory usage (assuming float32)
        stats['memory_usage_estimate'] = stats['total_parameters'] * 4  # bytes
        
        # Cache the statistics
        self.analysis_cache['statistics'] = stats
        
        logger.info("Attention statistics computed")
        return stats
    
    def export_analysis_report(
        self, 
        output_path: str,
        include_visualizations: bool = False
    ) -> str:
        """
        Export a comprehensive analysis report to JSON.
        
        Args:
            output_path: Path to save the analysis report
            include_visualizations: Whether to include visualization data
            
        Returns:
            Path to the saved report
        """
        logger.info(f"Exporting analysis report to {output_path}")
        
        # Collect all analysis data
        report = {
            'metadata': {
                'timestamp': pd.Timestamp.now().isoformat(),
                'total_layers': len(self.attention_maps),
                'analysis_version': '1.0'
            },
            'statistics': self.get_attention_statistics(),
            'attention_patterns': self.analyze_attention_patterns(),
            'token_influence': self.analyze_token_influence(prompt_length=50),  # Default prompt length
            'causal_attention': self.analyze_causal_attention_patterns(),
            'summary': self._create_analysis_summary()
        }
        
        # Save the report
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Analysis report saved to {output_path}")
        return str(output_path)
    
    def _create_analysis_summary(self) -> Dict[str, Any]:
        """Create a high-level summary of the attention analysis."""
        
        summary = {
            'key_insights': [],
            'attention_characteristics': {},
            'recommendations': []
        }
        
        # Get basic statistics
        stats = self.get_attention_statistics()
        
        # Key insights
        if stats['attention_weights_count'] > 0:
            summary['key_insights'].append(
                f"Captured attention weights from {stats['attention_weights_count']} attention operations"
            )
        
        if stats['layers_coverage']:
            summary['key_insights'].append(
                f"Covered {len(stats['layers_coverage'])} different layers/modules"
            )
        
        if stats['steps_coverage']:
            summary['key_insights'].append(
                f"Tracked attention across {len(stats['steps_coverage'])} generation steps"
            )
        
        # Attention characteristics
        if stats['total_parameters'] > 0:
            summary['attention_characteristics']['total_attention_parameters'] = stats['total_parameters']
            summary['attention_characteristics']['estimated_memory_usage_mb'] = stats['memory_usage_estimate'] / (1024 * 1024)
        
        # Recommendations
        if stats['attention_weights_count'] > 100:
            summary['recommendations'].append(
                "Consider downsampling attention weights for very long sequences to reduce memory usage"
            )
        
        if len(stats['layers_coverage']) > 20:
            summary['recommendations'].append(
                "Consider focusing on key layers for visualization to avoid overwhelming output"
            )
        
        return summary
    
    def get_layer_attention_summary(self, layer_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of attention patterns for a specific layer.
        
        Args:
            layer_name: Name of the layer to analyze
            
        Returns:
            Dictionary containing layer attention summary, or None if layer not found
        """
        if layer_name not in self.attention_maps:
            return None
        
        layer_data = self.attention_maps[layer_name]
        
        if not layer_data:
            return None
        
        summary = {
            'layer_name': layer_name,
            'total_captures': len(layer_data),
            'steps_covered': [],
            'attention_shapes': [],
            'attention_statistics': {}
        }
        
        # Collect attention weights for analysis
        attention_weights_list = []
        
        for data in layer_data:
            if 'step' in data:
                summary['steps_covered'].append(data['step'])
            
            if 'attention_weights' in data:
                summary['attention_shapes'].append(data['shape'])
                attention_weights_list.append(data['attention_weights'])
        
        # Analyze attention statistics
        if attention_weights_list:
            # Compute statistics across all captures
            all_attention = np.concatenate([aw.flatten() for aw in attention_weights_list])
            
            summary['attention_statistics'] = {
                'mean': float(np.mean(all_attention)),
                'std': float(np.std(all_attention)),
                'min': float(np.min(all_attention)),
                'max': float(np.max(all_attention)),
                'total_elements': int(len(all_attention))
            }
        
        summary['steps_covered'] = sorted(list(set(summary['steps_covered'])))
        
        return summary
    
    def analyze_causal_attention_patterns(self) -> Dict[str, Any]:
        """
        Analyze causal vs non-causal attention patterns across all layers.
        
        This method calculates the percentage of attention strength that is causal
        (tokens attending to previous tokens) vs non-causal (tokens attending to
        current/future tokens) for each layer, weighted by attention strength.
        
        Returns:
            Dictionary containing causal attention analysis per layer
        """
        if 'causal_attention' in self.analysis_cache:
            return self.analysis_cache['causal_attention']
        
        logger.info("Analyzing causal attention patterns...")
        
        analysis = {
            'layer_causal_analysis': {},
            'overall_summary': {},
            'step_evolution': {}
        }
        
        # Collect all unique layers and steps
        layers = set()
        steps = set()
        
        for module_name, data_list in self.attention_maps.items():
            layers.add(module_name)
            for data in data_list:
                if 'step' in data:
                    steps.add(data['step'])
        
        steps = sorted(list(steps))
        
        # Analyze each layer
        for layer_name in layers:
            layer_data = self.attention_maps[layer_name]
            
            if not layer_data:
                continue
            
            # Initialize layer analysis
            layer_analysis = {
                'total_causal_strength': 0.0,
                'total_non_causal_strength': 0.0,
                'causal_percentage': 0.0,
                'non_causal_percentage': 0.0,
                'mean_causal_strength': 0.0,
                'mean_non_causal_strength': 0.0,
                'step_analysis': {},
                'head_analysis': {}
            }
            
            # Collect attention weights for this layer
            attention_weights_list = []
            step_data = {}
            
            for data in layer_data:
                if 'attention_weights' in data:
                    step = data.get('step', 0)
                    step_data[step] = data['attention_weights']
                    attention_weights_list.append(data['attention_weights'])
            
            if not attention_weights_list:
                continue
            
            # Analyze each step for this layer
            for step, attn_weights in step_data.items():
                step_analysis = self._analyze_causal_attention_step(attn_weights, step)
                layer_analysis['step_analysis'][step] = step_analysis
                
                # Accumulate totals
                layer_analysis['total_causal_strength'] += step_analysis['causal_strength']
                layer_analysis['total_non_causal_strength'] += step_analysis['non_causal_strength']
            
            # Calculate overall percentages for this layer
            total_strength = layer_analysis['total_causal_strength'] + layer_analysis['total_non_causal_strength']
            if total_strength > 0:
                layer_analysis['causal_percentage'] = (layer_analysis['total_causal_strength'] / total_strength) * 100
                layer_analysis['non_causal_percentage'] = (layer_analysis['total_non_causal_strength'] / total_strength) * 100
                
                # Calculate mean strengths
                num_steps = len(layer_analysis['step_analysis'])
                layer_analysis['mean_causal_strength'] = layer_analysis['total_causal_strength'] / num_steps
                layer_analysis['mean_non_causal_strength'] = layer_analysis['total_non_causal_strength'] / num_steps
            
            # Analyze individual attention heads if available
            if attention_weights_list:
                layer_analysis['head_analysis'] = self._analyze_causal_attention_heads(attention_weights_list)
            
            analysis['layer_causal_analysis'][layer_name] = layer_analysis
        
        # Create overall summary
        analysis['overall_summary'] = self._create_causal_attention_summary(analysis['layer_causal_analysis'])
        
        # Analyze step evolution
        analysis['step_evolution'] = self._analyze_causal_attention_evolution(analysis['layer_causal_analysis'])
        
        # Cache the analysis
        self.analysis_cache['causal_attention'] = analysis
        
        logger.info("Causal attention analysis completed")
        return analysis
    
    def _analyze_causal_attention_step(self, attn_weights: np.ndarray, step: int) -> Dict[str, Any]:
        """
        Analyze causal attention patterns for a single step.
        
        Args:
            attn_weights: Attention weight matrix of shape [num_heads, seq_len, seq_len]
            step: Generation step number
            
        Returns:
            Dictionary containing step-level causal analysis
        """
        # Handle different tensor shapes
        if attn_weights.ndim == 4:
            # [batch, num_heads, seq_len, seq_len]
            attn_weights = attn_weights[0]  # Remove batch dimension
        elif attn_weights.ndim == 3:
            # [num_heads, seq_len, seq_len]
            pass
        else:
            logger.warning(f"Unexpected attention weights shape: {attn_weights.shape}")
            return {}
        
        num_heads, seq_len, _ = attn_weights.shape
        
        step_analysis = {
            'step': step,
            'sequence_length': seq_len,
            'num_heads': num_heads,
            'causal_strength': 0.0,
            'non_causal_strength': 0.0,
            'causal_percentage': 0.0,
            'non_causal_percentage': 0.0,
            'head_breakdown': {}
        }
        
        # Analyze each attention head
        for head_idx in range(num_heads):
            head_weights = attn_weights[head_idx]  # [seq_len, seq_len]
            
            # Calculate causal vs non-causal attention
            causal_strength = 0.0
            non_causal_strength = 0.0
            
            for i in range(seq_len):
                for j in range(seq_len):
                    if j < i:  # Causal: token i attends to previous token j
                        causal_strength += head_weights[i, j]
                    else:  # Non-causal: token i attends to current/future token j
                        non_causal_strength += head_weights[i, j]
            
            # Store head-level analysis
            total_head_strength = causal_strength + non_causal_strength
            head_percentage = (causal_strength / total_head_strength * 100) if total_head_strength > 0 else 0
            
            step_analysis['head_breakdown'][f'head_{head_idx}'] = {
                'causal_strength': float(causal_strength),
                'non_causal_strength': float(non_causal_strength),
                'causal_percentage': float(head_percentage),
                'non_causal_percentage': float(100 - head_percentage)
            }
            
            # Accumulate total strengths
            step_analysis['causal_strength'] += causal_strength
            step_analysis['non_causal_strength'] += non_causal_strength
        
        # Calculate overall percentages for this step
        total_step_strength = step_analysis['causal_strength'] + step_analysis['non_causal_strength']
        if total_step_strength > 0:
            step_analysis['causal_percentage'] = (step_analysis['causal_strength'] / total_step_strength) * 100
            step_analysis['non_causal_percentage'] = (step_analysis['non_causal_strength'] / total_step_strength) * 100
        
        return step_analysis
    
    def _analyze_causal_attention_heads(self, attention_weights_list: List[np.ndarray]) -> Dict[str, Any]:
        """
        Analyze causal attention patterns across all heads for a layer.
        
        Args:
            attention_weights_list: List of attention weight matrices for this layer
            
        Returns:
            Dictionary containing head-level causal analysis
        """
        head_analysis = {}
        
        # Get the first attention matrix to determine number of heads
        if not attention_weights_list:
            return head_analysis
        
        first_weights = attention_weights_list[0]
        if first_weights.ndim == 4:
            num_heads = first_weights.shape[1]
        elif first_weights.ndim == 3:
            num_heads = first_weights.shape[0]
        else:
            return head_analysis
        
        # Initialize head analysis
        for head_idx in range(num_heads):
            head_analysis[f'head_{head_idx}'] = {
                'total_causal_strength': 0.0,
                'total_non_causal_strength': 0.0,
                'causal_percentage': 0.0,
                'non_causal_percentage': 0.0,
                'steps_analyzed': 0
            }
        
        # Analyze each step
        for attn_weights in attention_weights_list:
            if attn_weights.ndim == 4:
                attn_weights = attn_weights[0]  # Remove batch dimension
            elif attn_weights.ndim == 3:
                pass
            else:
                continue
            
            num_heads, seq_len, _ = attn_weights.shape
            
            for head_idx in range(num_heads):
                head_weights = attn_weights[head_idx]
                
                # Calculate causal vs non-causal attention for this head
                causal_strength = 0.0
                non_causal_strength = 0.0
                
                for i in range(seq_len):
                    for j in range(seq_len):
                        if j < i:  # Causal
                            causal_strength += head_weights[i, j]
                        else:  # Non-causal
                            non_causal_strength += head_weights[i, j]
                
                # Accumulate for this head
                head_key = f'head_{head_idx}'
                if head_key in head_analysis:
                    head_analysis[head_key]['total_causal_strength'] += causal_strength
                    head_analysis[head_key]['total_non_causal_strength'] += non_causal_strength
                    head_analysis[head_key]['steps_analyzed'] += 1
        
        # Calculate percentages for each head
        for head_data in head_analysis.values():
            total_strength = head_data['total_causal_strength'] + head_data['total_non_causal_strength']
            if total_strength > 0:
                head_data['causal_percentage'] = (head_data['total_causal_strength'] / total_strength) * 100
                head_data['non_causal_percentage'] = (head_data['total_non_causal_strength'] / total_strength) * 100
        
        return head_analysis
    
    def _create_causal_attention_summary(self, layer_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary of causal attention patterns across all layers.
        
        Args:
            layer_analysis: Dictionary containing layer-level causal analysis
            
        Returns:
            Dictionary containing overall summary
        """
        if not layer_analysis:
            return {}
        
        summary = {
            'total_layers': len(layer_analysis),
            'overall_causal_percentage': 0.0,
            'overall_non_causal_percentage': 0.0,
            'layer_rankings': {},
            'causal_distribution': {
                'highly_causal': [],      # >80% causal
                'moderately_causal': [],  # 50-80% causal
                'balanced': [],           # 40-60% causal
                'moderately_non_causal': [], # 20-40% causal
                'highly_non_causal': []   # <20% causal
            }
        }
        
        # Calculate overall percentages
        total_causal = sum(layer['total_causal_strength'] for layer in layer_analysis.values())
        total_non_causal = sum(layer['total_non_causal_strength'] for layer in layer_analysis.values())
        total_strength = total_causal + total_non_causal
        
        if total_strength > 0:
            summary['overall_causal_percentage'] = (total_causal / total_strength) * 100
            summary['overall_non_causal_percentage'] = (total_non_causal / total_strength) * 100
        
        # Categorize layers by causal percentage
        for layer_name, layer_data in layer_analysis.items():
            causal_pct = layer_data['causal_percentage']
            
            if causal_pct > 80:
                summary['causal_distribution']['highly_causal'].append(layer_name)
            elif causal_pct > 50:
                summary['causal_distribution']['moderately_causal'].append(layer_name)
            elif causal_pct > 40:
                summary['causal_distribution']['balanced'].append(layer_name)
            elif causal_pct > 20:
                summary['causal_distribution']['moderately_non_causal'].append(layer_name)
            else:
                summary['causal_distribution']['highly_non_causal'].append(layer_name)
        
        # Create layer rankings by causal percentage
        layer_rankings = sorted(
            [(name, data['causal_percentage']) for name, data in layer_analysis.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        summary['layer_rankings'] = {
            'most_causal': [name for name, _ in layer_rankings[:5]],
            'least_causal': [name for name, _ in layer_rankings[-5:]]
        }
        
        return summary
    
    def get_causal_attention_summary(self) -> Dict[str, Any]:
        """
        Get a quick summary of causal attention patterns.
        
        Returns:
            Dictionary containing causal attention summary
        """
        causal_analysis = self.analyze_causal_attention_patterns()
        return causal_analysis.get('overall_summary', {})
    
    def get_layer_causal_percentages(self) -> Dict[str, float]:
        """
        Get causal attention percentages for each layer.
        
        Returns:
            Dictionary mapping layer names to causal percentages
        """
        causal_analysis = self.analyze_causal_attention_patterns()
        layer_analysis = causal_analysis.get('layer_causal_analysis', {})
        
        return {
            layer_name: layer_data['causal_percentage']
            for layer_name, layer_data in layer_analysis.items()
        }
    
    def _analyze_causal_attention_evolution(self, layer_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze how causal attention patterns evolve across generation steps.
        
        Args:
            layer_analysis: Dictionary containing layer-level causal analysis
            
        Returns:
            Dictionary containing step evolution analysis
        """
        evolution = {
            'step_trends': {},
            'layer_consistency': {},
            'overall_evolution': {}
        }
        
        # Analyze step-by-step evolution
        all_steps = set()
        for layer_data in layer_analysis.values():
            for step in layer_data['step_analysis'].keys():
                all_steps.add(step)
        
        all_steps = sorted(list(all_steps))
        
        # Calculate average causal percentage per step
        step_causal_percentages = {}
        for step in all_steps:
            step_percentages = []
            for layer_data in layer_analysis.values():
                if step in layer_data['step_analysis']:
                    step_percentages.append(layer_data['step_analysis'][step]['causal_percentage'])
            
            if step_percentages:
                step_causal_percentages[step] = {
                    'mean': np.mean(step_percentages),
                    'std': np.std(step_percentages),
                    'min': np.min(step_percentages),
                    'max': np.max(step_percentages)
                }
        
        evolution['step_trends'] = step_causal_percentages
        
        # Analyze layer consistency across steps
        for layer_name, layer_data in layer_analysis.items():
            step_percentages = []
            for step_data in layer_data['step_analysis'].values():
                step_percentages.append(step_data['causal_percentage'])
            
            if step_percentages:
                evolution['layer_consistency'][layer_name] = {
                    'mean': np.mean(step_percentages),
                    'std': np.std(step_percentages),
                    'min': np.min(step_percentages),
                    'max': np.max(step_percentages),
                    'consistency_score': 1.0 / (1.0 + np.std(step_percentages))  # Higher = more consistent
                }
        
        # Overall evolution summary
        if step_causal_percentages:
            all_step_means = [data['mean'] for data in step_causal_percentages.values()]
            evolution['overall_evolution'] = {
                'trend': 'increasing' if all_step_means[-1] > all_step_means[0] else 'decreasing',
                'total_change': all_step_means[-1] - all_step_means[0],
                'volatility': np.std(all_step_means)
            }
        
        return evolution
