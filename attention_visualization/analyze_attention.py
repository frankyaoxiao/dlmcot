#!/usr/bin/env python3
"""
Comprehensive Attention Analysis Script for MMaDA Models

This script analyzes attention data captured during text generation and creates
various visualizations and analysis reports.
"""

import argparse
import sys
import logging
from pathlib import Path
import json
import torch
import numpy as np

# Add the parent directory to the path to import mmada_inference
sys.path.insert(0, str(Path(__file__).parent.parent))

from attention_visualization.attention_visualizer import AttentionVisualizer
from attention_visualization.attention_analyzer import AttentionAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_attention_data(data_path: str):
    """
    Load attention data from various sources.
    
    Args:
        data_path: Path to attention data (JSON file, pickle file, or directory)
        
    Returns:
        Dictionary containing attention maps
    """
    data_path = Path(data_path)
    
    if data_path.is_file():
        if data_path.suffix == '.json':
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            # Check if this is a generation history file with attention maps
            if 'attention_maps' in data:
                return data['attention_maps']
            else:
                return data
        else:
            logger.error(f"Unsupported file format: {data_path.suffix}")
            return None
    elif data_path.is_dir():
        # Look for attention data files in the directory
        attention_files = list(data_path.glob("*attention*.json"))
        if not attention_files:
            logger.error(f"No attention data files found in {data_path}")
            return None
        
        # Load the first attention file found
        logger.info(f"Loading attention data from {attention_files[0]}")
        with open(attention_files[0], 'r') as f:
            data = json.load(f)
        
        if 'attention_maps' in data:
            return data['attention_maps']
        else:
            return data
    else:
        logger.error(f"Path does not exist: {data_path}")
        return None

def create_attention_visualizations(
    attention_maps: dict,
    output_dir: str,
    prompt_tokens: list = None,
    generated_tokens: list = None,
    max_layers: int = 16,
    max_tokens: int = 100
):
    """
    Create comprehensive attention visualizations.
    
    Args:
        attention_maps: Dictionary of attention maps from AttentionHook
        output_dir: Directory to save visualizations
        prompt_tokens: List of prompt token texts
        generated_tokens: List of generated token texts
        max_layers: Maximum number of layers to display
        max_tokens: Maximum number of tokens to analyze
    """
    logger.info("Creating attention visualizations...")
    
    # Create visualizer
    visualizer = AttentionVisualizer(output_dir=output_dir)
    
    try:
        # Create comprehensive report
        saved_files = visualizer.create_comprehensive_attention_report(
            attention_maps,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens
        )
        
        logger.info(f"Created {len(saved_files)} visualization files")
        
        # Print summary of created files
        for viz_type, filepath in saved_files.items():
            logger.info(f"  {viz_type}: {filepath}")
        
        return saved_files
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return {}

def analyze_attention_patterns(
    attention_maps: dict,
    output_dir: str,
    prompt_length: int = 50
):
    """
    Analyze attention patterns and create analysis report.
    
    Args:
        attention_maps: Dictionary of attention maps from AttentionHook
        output_dir: Directory to save analysis report
        prompt_length: Length of the prompt for token influence analysis
    """
    logger.info("Analyzing attention patterns...")
    
    # Create analyzer
    analyzer = AttentionAnalyzer(attention_maps)
    
    try:
        # Get basic statistics
        stats = analyzer.get_attention_statistics()
        logger.info(f"Attention statistics: {stats['total_attention_maps']} maps, {stats['attention_weights_count']} weights")
        
        # Analyze attention patterns
        patterns = analyzer.analyze_attention_patterns()
        logger.info(f"Analyzed {patterns['total_layers']} layers across {patterns['total_steps']} steps")
        
        # Analyze token influence
        token_influence = analyzer.analyze_token_influence(prompt_length=prompt_length)
        logger.info("Token influence analysis completed")
        
        # Export comprehensive report
        report_path = Path(output_dir) / "attention_analysis_report.json"
        analyzer.export_analysis_report(str(report_path))
        logger.info(f"Analysis report exported to {report_path}")
        
        return {
            'statistics': stats,
            'patterns': patterns,
            'token_influence': token_influence,
            'report_path': str(report_path)
        }
        
    except Exception as e:
        logger.error(f"Error analyzing attention patterns: {e}")
        import traceback
        traceback.print_exc()
        return {}

def create_token_analysis(
    attention_maps: dict,
    output_dir: str,
    prompt_length: int = 50
):
    """
    Create detailed token-level analysis.
    
    Args:
        attention_maps: Dictionary of attention maps from AttentionHook
        output_dir: Directory to save analysis
        prompt_length: Length of the prompt
    """
    logger.info("Creating token-level analysis...")
    
    try:
        # Create analyzer
        analyzer = AttentionAnalyzer(attention_maps)
        
        # Get token influence analysis
        token_influence = analyzer.analyze_token_influence(prompt_length=prompt_length)
        
        # Create detailed token analysis report
        token_analysis = {
            'prompt_tokens': {},
            'generated_tokens': {},
            'cross_attention': token_influence.get('cross_token_influence_summary', {})
        }
        
        # Analyze prompt tokens
        for token_idx, token_data in token_influence.get('prompt_tokens', {}).items():
            token_analysis['prompt_tokens'][f"token_{token_idx}"] = {
                'position': token_idx,
                'avg_outgoing_influence': token_data.get('avg_outgoing', 0),
                'avg_incoming_influence': token_data.get('avg_incoming', 0),
                'total_steps': token_data.get('total_steps', 0),
                'total_layers': token_data.get('total_layers', 0)
            }
        
        # Analyze generated tokens
        for token_idx, token_data in token_influence.get('generated_tokens', {}).items():
            token_analysis['generated_tokens'][f"token_{token_idx}"] = {
                'position': token_idx,
                'avg_outgoing_influence': token_data.get('avg_outgoing', 0),
                'avg_incoming_influence': token_data.get('avg_incoming', 0),
                'total_steps': token_data.get('total_steps', 0),
                'total_layers': token_data.get('total_layers', 0)
            }
        
        # Save token analysis
        token_analysis_path = Path(output_dir) / "token_analysis.json"
        with open(token_analysis_path, 'w') as f:
            json.dump(token_analysis, f, indent=2, default=str)
        
        logger.info(f"Token analysis saved to {token_analysis_path}")
        return token_analysis
        
    except Exception as e:
        logger.error(f"Error creating token analysis: {e}")
        import traceback
        traceback.print_exc()
        return {}

def main():
    """Main function to run attention analysis."""
    
    parser = argparse.ArgumentParser(
        description="Analyze attention patterns from MMaDA generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze attention data from a generation history file
  python analyze_attention.py --input logs/generation_with_attention.json --output attention_analysis
  
  # Analyze with custom prompt length
  python analyze_attention.py --input attention_data.json --output analysis --prompt-length 100
  
  # Create visualizations with limited layers
  python analyze_attention.py --input attention_data.json --output analysis --max-layers 8
        """
    )
    
    parser.add_argument(
        "--input", "-i", 
        required=True,
        help="Path to attention data file or directory"
    )
    
    parser.add_argument(
        "--output", "-o", 
        default="attention_analysis",
        help="Output directory for analysis and visualizations (default: attention_analysis)"
    )
    
    parser.add_argument(
        "--prompt-length", "-p",
        type=int,
        default=50,
        help="Length of the prompt for token analysis (default: 50)"
    )
    
    parser.add_argument(
        "--max-layers", "-l",
        type=int,
        default=16,
        help="Maximum number of layers to visualize (default: 16)"
    )
    
    parser.add_argument(
        "--max-tokens", "-t",
        type=int,
        default=100,
        help="Maximum number of tokens to analyze (default: 100)"
    )
    
    parser.add_argument(
        "--no-visualizations", "-nv",
        action="store_true",
        help="Skip creating visualizations (analysis only)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=" * 60)
    logger.info("MMaDA ATTENTION ANALYSIS")
    logger.info("=" * 60)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Load attention data
    logger.info(f"Loading attention data from: {args.input}")
    attention_maps = load_attention_data(args.input)
    
    if attention_maps is None:
        logger.error("Failed to load attention data")
        sys.exit(1)
    
    logger.info(f"Loaded attention data with {len(attention_maps)} modules")
    
    # Create visualizations (unless disabled)
    if not args.no_visualizations:
        logger.info("Creating attention visualizations...")
        viz_files = create_attention_visualizations(
            attention_maps,
            str(output_dir),
            max_layers=args.max_layers,
            max_tokens=args.max_tokens
        )
        
        if viz_files:
            logger.info(f"✓ Created {len(viz_files)} visualization files")
        else:
            logger.warning("No visualizations were created")
    
    # Analyze attention patterns
    logger.info("Analyzing attention patterns...")
    analysis_results = analyze_attention_patterns(
        attention_maps,
        str(output_dir),
        prompt_length=args.prompt_length
    )
    
    if analysis_results:
        logger.info("✓ Attention pattern analysis completed")
    else:
        logger.warning("Attention pattern analysis failed")
    
    # Create token analysis
    logger.info("Creating token-level analysis...")
    token_analysis = create_token_analysis(
        attention_maps,
        str(output_dir),
        prompt_length=args.prompt_length
    )
    
    if token_analysis:
        logger.info("✓ Token analysis completed")
    else:
        logger.warning("Token analysis failed")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS COMPLETED")
    logger.info("=" * 60)
    
    logger.info(f"Output directory: {output_dir}")
    
    if not args.no_visualizations:
        logger.info("Visualizations: ✓ Created")
    else:
        logger.info("Visualizations: ✗ Skipped")
    
    if analysis_results:
        logger.info("Pattern Analysis: ✓ Completed")
    else:
        logger.info("Pattern Analysis: ✗ Failed")
    
    if token_analysis:
        logger.info("Token Analysis: ✓ Completed")
    else:
        logger.info("Token Analysis: ✗ Failed")
    
    logger.info("\n🎉 Attention analysis completed successfully!")
    logger.info(f"Check the output directory '{output_dir}' for results.")

if __name__ == "__main__":
    main()
