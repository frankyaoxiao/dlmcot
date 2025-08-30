"""
Attention Visualizer for MMaDA Models

This module provides functionality to create attention heatmaps and visualizations
from captured attention weights during text generation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import os
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class AttentionVisualizer:
    """
    Creates attention visualizations from captured attention weights.
    
    This class generates various types of attention visualizations including:
    - Per-layer attention heatmaps
    - Attention evolution across generation steps
    - Block-level attention patterns
    - Token-level attention analysis
    """
    
    def __init__(self, output_dir: str = "attention_visualizations"):
        """
        Initialize the attention visualizer.
        
        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up seaborn style for better aesthetics
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        
        logger.info(f"AttentionVisualizer initialized with output directory: {self.output_dir}")
    
    def create_layer_attention_comparison(
        self,
        attention_maps: Dict[str, List[Dict[str, Any]]],
        step: int,
        prompt_tokens: Optional[List[str]] = None,
        generated_tokens: Optional[List[str]] = None,
        max_layers: int = 32,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a comparison of attention patterns across multiple layers for a specific step.
        Now creates one figure per layer to show all 32 heads properly.
        
        Args:
            attention_maps: Dictionary of attention maps from AttentionHook
            step: Generation step to visualize
            prompt_tokens: List of prompt token texts
            generated_tokens: List of generated token texts
            max_layers: Maximum number of layers to display
            save_path: Optional path to save the visualization
            
        Returns:
            List of matplotlib Figure objects (one per layer)
        """
        try:
            import seaborn as sns
            
            # Set seaborn style for better aesthetics
            sns.set_style("whitegrid")
            
            # Use consistent dark red color scheme
            color_palette = 'Reds'
            
            # Filter attention maps for the specific step and get main transformer blocks only
            step_maps = {}
            for module_name, data_list in attention_maps.items():
                # Only include main transformer blocks (not individual projections)
                if ('blocks.' in module_name and 
                    not any(proj in module_name for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj'])):
                    for data in data_list:
                        if data.get('step') == step and 'attention_weights' in data:
                            step_maps[module_name] = data['attention_weights']
                            break
            
            if not step_maps:
                logger.warning(f"No attention maps found for step {step}")
                return []
            
            # Sort by layer number and limit the number of layers
            sorted_modules = sorted(step_maps.keys(), key=lambda x: self._extract_layer_number(x))
            if len(sorted_modules) > max_layers:
                sorted_modules = sorted_modules[:max_layers]
                logger.info(f"Limited visualization to {max_layers} layers")
            
            figures = []
            
            # Create one figure per layer
            for layer_idx, module_name in enumerate(sorted_modules):
                attention_weights = step_maps[module_name]
                layer_num = self._extract_layer_number(module_name)
                
                # Handle different tensor shapes
                if attention_weights.ndim == 4:
                    # Shape: [batch, heads, seq_len, seq_len]
                    attention_weights = attention_weights[0]  # Remove batch dimension
                elif attention_weights.ndim == 3:
                    # Shape: [heads, seq_len, seq_len]
                    pass
                else:
                    # Single attention matrix
                    attention_weights = attention_weights.unsqueeze(0)
                
                # Get number of attention heads
                num_heads = attention_weights.shape[0]
                
                # Limit tokens for readability
                max_tokens = 25
                if attention_weights.shape[-1] > max_tokens:
                    attention_weights = attention_weights[:, :max_tokens, :max_tokens]
                
                # Create subplot grid for all heads in this layer
                # Use 8x4 grid for 32 heads (or adjust based on actual number)
                n_heads = min(num_heads, 32)
                cols = 8
                rows = (n_heads + cols - 1) // cols
                
                # Create figure with proper spacing
                fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 2.5*rows))
                
                # Handle single row case
                if rows == 1:
                    axes = axes.reshape(1, -1)
                if cols == 1:
                    axes = axes.reshape(-1, 1)
                
                # Plot each attention head
                for head_idx in range(n_heads):
                    row_idx = head_idx // cols
                    col_idx = head_idx % cols
                    ax = axes[row_idx, col_idx]
                    
                    # Get attention weights for this head
                    if head_idx < attention_weights.shape[0]:
                        head_weights = attention_weights[head_idx]
                    else:
                        # If we don't have enough heads, use the first one
                        head_weights = attention_weights[0]
                    
                    # Create seaborn heatmap with consistent red color scheme
                    sns.heatmap(
                        head_weights,
                        ax=ax,
                        cmap=color_palette,
                        cbar=False,  # No colorbar to avoid overlapping
                        square=True,
                        xticklabels=False,
                        yticklabels=False,
                        vmin=0,
                        vmax=head_weights.max()
                    )
                    
                    # Set title for each head
                    ax.set_title(f'Head {head_idx}', fontsize=8, pad=5)
                    
                    # Highlight prompt boundary if available
                    if prompt_tokens and len(prompt_tokens) <= max_tokens:
                        prompt_end = len(prompt_tokens) - 0.5
                        if prompt_end < max_tokens:
                            ax.axvline(x=prompt_end, color='red', linestyle='--', alpha=0.8, linewidth=1.5)
                            ax.axhline(y=prompt_end, color='red', linestyle='--', alpha=0.8, linewidth=1.5)
                
                # Hide unused subplots
                for i in range(n_heads, rows * cols):
                    row_idx = i // cols
                    col_idx = i % cols
                    axes[row_idx, col_idx].set_visible(False)
                
                # Add simple title with just layer and step numbers
                fig.suptitle(f'Layer {layer_num} - Step {step}', 
                            fontsize=16, fontweight='bold', y=0.98)
                
                # Adjust layout to prevent overlapping
                plt.tight_layout()
                plt.subplots_adjust(top=0.92, bottom=0.05, left=0.05, right=0.95, hspace=0.3, wspace=0.3)
                
                figures.append(fig)
            
            # Save if path provided
            if save_path:
                # Create directory structure: step_folder/layer_files
                step_dir = Path(save_path).parent / f"step_{step}"
                step_dir.mkdir(exist_ok=True)
                
                for layer_idx, fig in enumerate(figures):
                    layer_num = self._extract_layer_number(sorted_modules[layer_idx])
                    layer_filename = step_dir / f"layer_{layer_num}_all_heads.png"
                    fig.savefig(layer_filename, dpi=300, bbox_inches='tight', facecolor='white')
                    plt.close(fig)
                    logger.info(f"Saved layer {layer_num} to: {layer_filename}")
                
                return figures
            
            return figures
            
        except Exception as e:
            logger.error(f"Error creating layer attention comparison: {e}")
            raise
    
    def create_attention_evolution_plot(
        self,
        attention_maps: Dict[str, List[Dict[str, Any]]],
        layer_name: str,
        prompt_tokens: Optional[List[str]] = None,
        generated_tokens: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a plot showing how attention patterns evolve across generation steps.
        
        Args:
            attention_maps: Dictionary of attention maps from AttentionHook
            layer_name: Specific layer to visualize evolution for
            prompt_tokens: List of prompt token texts
            generated_tokens: List of generated token texts
            save_path: Optional path to save the visualization
            
        Returns:
            matplotlib Figure object
        """
        try:
            # Get attention data for the specific layer
            if layer_name not in attention_maps:
                logger.warning(f"Layer {layer_name} not found in attention maps")
                return None
            
            layer_data = attention_maps[layer_name]
            
            # Sort by step
            layer_data.sort(key=lambda x: x.get('step', 0))
            
            # Extract attention weights and steps
            steps = []
            attention_weights_list = []
            
            for data in layer_data:
                if 'attention_weights' in data:
                    steps.append(data.get('step', 0))
                    attention_weights_list.append(data['attention_weights'])
            
            if not attention_weights_list:
                logger.warning(f"No attention weights found for layer {layer_name}")
                return None
            
            # Create the figure with seaborn style
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Attention weights over time (averaged across heads)
            ax1 = axes[0, 0]
            for i, (step, attn_weights) in enumerate(zip(steps, attention_weights_list)):
                if attn_weights.ndim == 4:
                    attn_weights = attn_weights[0].mean(axis=0)  # Average across heads
                elif attn_weights.ndim == 3:
                    attn_weights = attn_weights.mean(axis=0)  # Average across heads
                
                # Plot attention weights as a line (averaged across sequence)
                avg_attention = attn_weights.mean(axis=1)
                ax1.plot(avg_attention, label=f'Step {step}', alpha=0.7, linewidth=1)
            
            ax1.set_xlabel('Token Position', fontsize=12)
            ax1.set_ylabel('Average Attention Weight', fontsize=12)
            ax1.set_title('Attention Evolution Across Steps', fontsize=14, fontweight='bold')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Attention heatmap for first step
            ax2 = axes[0, 1]
            if attention_weights_list:
                first_attn = attention_weights_list[0]
                if first_attn.ndim == 4:
                    first_attn = first_attn[0].mean(axis=0)
                elif first_attn.ndim == 3:
                    first_attn = first_attn.mean(axis=0)
                
                # Limit tokens for readability
                max_tokens = 40
                if first_attn.shape[0] > max_tokens:
                    first_attn = first_attn[:max_tokens, :max_tokens]
                
                sns.heatmap(first_attn, ax=ax2, cmap='Blues', square=True, xticklabels=False, yticklabels=False)
                ax2.set_title(f'Attention Pattern - Step {steps[0]}', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Key Tokens', fontsize=12)
                ax2.set_ylabel('Query Tokens', fontsize=12)
                
                # Highlight prompt boundary
                if prompt_tokens and len(prompt_tokens) <= max_tokens:
                    prompt_end = len(prompt_tokens) - 0.5
                    if prompt_end < max_tokens:
                        ax2.axvline(x=prompt_end, color='red', linestyle='--', alpha=0.8, linewidth=2)
                        ax2.axhline(y=prompt_end, color='red', linestyle='--', alpha=0.8, linewidth=2)
            
            # Plot 3: Attention heatmap for last step
            ax3 = axes[1, 0]
            if attention_weights_list:
                last_attn = attention_weights_list[-1]
                if last_attn.ndim == 4:
                    last_attn = last_attn[0].mean(axis=0)
                elif last_attn.ndim == 3:
                    last_attn = last_attn.mean(axis=0)
                
                # Limit tokens for readability
                max_tokens = 40
                if last_attn.shape[0] > max_tokens:
                    last_attn = last_attn[:max_tokens, :max_tokens]
                
                sns.heatmap(last_attn, ax=ax3, cmap='Blues', square=True, xticklabels=False, yticklabels=False)
                ax3.set_title(f'Attention Pattern - Step {steps[-1]}', fontsize=14, fontweight='bold')
                ax3.set_xlabel('Key Tokens', fontsize=12)
                ax3.set_ylabel('Query Tokens', fontsize=12)
                
                # Highlight prompt boundary
                if prompt_tokens and len(prompt_tokens) <= max_tokens:
                    prompt_end = len(prompt_tokens) - 0.5
                    if prompt_end < max_tokens:
                        ax3.axvline(x=prompt_end, color='red', linestyle='--', alpha=0.8, linewidth=2)
                        ax3.axhline(y=prompt_end, color='red', linestyle='--', alpha=0.8, linewidth=2)
            
            # Plot 4: Attention change between first and last step
            ax4 = axes[1, 1]
            if len(attention_weights_list) >= 2:
                first_attn = attention_weights_list[0]
                last_attn = attention_weights_list[-1]
                
                if first_attn.ndim == 4:
                    first_attn = first_attn[0].mean(axis=0)
                    last_attn = last_attn[0].mean(axis=0)
                elif first_attn.ndim == 3:
                    first_attn = first_attn.mean(axis=0)
                    last_attn = last_attn.mean(axis=0)
                
                # Limit tokens for readability
                max_tokens = 40
                if first_attn.shape[0] > max_tokens:
                    first_attn = first_attn[:max_tokens, :max_tokens]
                    last_attn = last_attn[:max_tokens, :max_tokens]
                
                # Compute attention change
                attn_change = last_attn - first_attn
                
                sns.heatmap(attn_change, ax=ax4, cmap='RdBu_r', square=True, xticklabels=False, yticklabels=False, center=0)
                ax4.set_title('Attention Change (Last - First Step)', fontsize=14, fontweight='bold')
                ax4.set_xlabel('Key Tokens', fontsize=12)
                ax4.set_ylabel('Query Tokens', fontsize=12)
                
                # Highlight prompt boundary
                if prompt_tokens and len(prompt_tokens) <= max_tokens:
                    prompt_end = len(prompt_tokens) - 0.5
                    if prompt_end < max_tokens:
                        ax4.axvline(x=prompt_end, color='red', linestyle='--', alpha=0.8, linewidth=2)
                        ax4.axhline(y=prompt_end, color='red', linestyle='--', alpha=0.8, linewidth=2)
            
            # Overall title
            layer_num = self._extract_layer_number(layer_name)
            fig.suptitle(f'Attention Evolution for Layer {layer_num}', fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                logger.info(f"Saved attention evolution plot to: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating attention evolution plot: {e}")
            raise
    
    def create_comprehensive_attention_report(
        self,
        attention_maps: Dict[str, List[Dict[str, Any]]],
        prompt_tokens: Optional[List[str]] = None,
        generated_tokens: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        max_layers: int = 8
    ) -> Dict[str, str]:
        """
        Create a comprehensive attention visualization report.
        
        Args:
            attention_maps: Dictionary of attention maps from AttentionHook
            prompt_tokens: List of prompt token texts
            generated_tokens: List of generated token texts
            output_dir: Directory to save the report (uses self.output_dir if None)
            max_layers: Maximum number of layers to display
            
        Returns:
            Dictionary mapping visualization type to saved file path
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        saved_files = {}
        
        try:
            # 1. Create layer comparison for each step (limit to key steps)
            steps = set()
            for module_name, data_list in attention_maps.items():
                for data in data_list:
                    if 'step' in data:
                        steps.add(data['step'])
            
            # Only create plots for key steps to avoid too many files
            key_steps = sorted(steps)[::64]  # Every 64th step to reduce file count
            if len(key_steps) > 4:  # Limit to 4 key steps
                key_steps = key_steps[:4]
            
            for step in key_steps:
                try:
                    figures = self.create_layer_attention_comparison(
                        attention_maps, step, prompt_tokens, generated_tokens, max_layers,
                        save_path=str(output_dir / f"step_{step}_placeholder.png")
                    )
                    if figures:
                        # The method now creates its own directory structure
                        step_dir = output_dir / f"step_{step}"
                        saved_files[f"layer_comparison_step_{step}"] = str(step_dir)
                        logger.info(f"Saved layer comparison for step {step} with {len(figures)} layers")
                except Exception as e:
                    logger.warning(f"Failed to create layer comparison for step {step}: {e}")
            
            # 2. Create attention evolution plots for key layers
            key_layers = []
            for module_name in attention_maps.keys():
                # Look for main transformer blocks (not individual projections)
                if ('blocks.' in module_name and 
                    not any(proj in module_name for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']) and
                    len(attention_maps[module_name]) > 1):
                    key_layers.append(module_name)
            
            # Limit to first few layers for readability
            key_layers = key_layers[:8]
            
            for layer_name in key_layers:
                try:
                    fig = self.create_attention_evolution_plot(
                        attention_maps, layer_name, prompt_tokens, generated_tokens
                    )
                    if fig:
                        filename = f"attention_evolution_{layer_name.replace('/', '_')}_{timestamp}.png"
                        filepath = output_dir / filename
                        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
                        plt.close(fig)
                        saved_files[f"attention_evolution_{layer_name}"] = str(filepath)
                        logger.info(f"Saved attention evolution for {layer_name}")
                except Exception as e:
                    logger.warning(f"Failed to create attention evolution for {layer_name}: {e}")
            
            # 3. Create summary statistics
            summary_stats = self._create_attention_summary(attention_maps)
            summary_file = output_dir / f"attention_summary_{timestamp}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary_stats, f, indent=2, default=str)
            saved_files["summary_stats"] = str(summary_file)
            
            logger.info(f"Comprehensive attention report created with {len(saved_files)} visualizations")
            
        except Exception as e:
            logger.error(f"Error creating comprehensive attention report: {e}")
            raise
        
        return saved_files
    
    def _extract_layer_number(self, module_name: str) -> int:
        """Extract layer number from module name for sorting."""
        try:
            # Look for patterns like "blocks.0", "blocks.1", etc.
            import re
            match = re.search(r'blocks\.(\d+)', module_name)
            if match:
                return int(match.group(1))
            return 0
        except:
            return 0
    
    def _create_attention_summary(self, attention_maps: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Create summary statistics for attention maps."""
        summary = {
            'total_modules': len(attention_maps),
            'total_attention_captures': 0,
            'steps_covered': set(),
            'layers_with_attention': [],
            'attention_shapes': [],
            'generation_metadata': {}
        }
        
        for module_name, data_list in attention_maps.items():
            summary['total_attention_captures'] += len(data_list)
            
            for data in data_list:
                if 'step' in data:
                    summary['steps_covered'].add(data['step'])
                
                if 'attention_weights' in data:
                    summary['layers_with_attention'].append(module_name)
                    summary['attention_shapes'].append(data['shape'])
                
                if 'step' in data and 'block' in data:
                    step_block = f"step_{data['step']}_block_{data['block']}"
                    if step_block not in summary['generation_metadata']:
                        summary['generation_metadata'][step_block] = {
                            'modules_captured': [],
                            'attention_weights_count': 0
                        }
                    summary['generation_metadata'][step_block]['modules_captured'].append(module_name)
                    if 'attention_weights' in data:
                        summary['generation_metadata'][step_block]['attention_weights_count'] += 1
        
        # Convert sets to lists for JSON serialization
        summary['steps_covered'] = sorted(list(summary['steps_covered']))
        
        return summary
