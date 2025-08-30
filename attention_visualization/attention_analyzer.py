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
