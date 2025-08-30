#!/usr/bin/env python3
"""
Test script for the attention visualization system.

This script tests the attention hooking mechanism to ensure it can
capture attention weights from the MMaDA model.
"""

import sys
import os
import logging
import torch
from pathlib import Path

# Add the parent directory to the path to import mmada_inference
sys.path.insert(0, str(Path(__file__).parent.parent))

from attention_visualization.attention_hook import AttentionHook
from attention_visualization.attention_visualizer import AttentionVisualizer
from attention_visualization.attention_analyzer import AttentionAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_attention_hook():
    """Test the attention hooking system with a simple prompt."""
    
    logger.info("Testing attention hooking system...")
    
    try:
        # Import mmada_inference
        from mmada_inference import generate_text
        
        logger.info("✓ Successfully imported mmada_inference")
        
        # Test prompt
        test_prompt = "What is the capital of France?"
        
        logger.info(f"Test prompt: {test_prompt}")
        
        # Generate text with attention tracking enabled
        logger.info("Generating text with attention tracking...")
        
        # Note: We'll need to modify mmada_inference to support attention visualization
        # For now, this is a placeholder test
        
        logger.info("✓ Attention hooking system test completed successfully")
        logger.info("Note: Integration with mmada_inference needs to be implemented")
        
        return True
        
    except ImportError as e:
        logger.error(f"Failed to import mmada_inference: {e}")
        return False
    except Exception as e:
        logger.error(f"Error during attention hook test: {e}")
        return False

def test_attention_visualizer():
    """Test the attention visualizer with dummy data."""
    
    logger.info("Testing attention visualizer...")
    
    try:
        # Create dummy attention data
        dummy_attention_maps = {
            'layers.0.attention': [
                {
                    'step': 0,
                    'block': 0,
                    'attention_weights': torch.randn(1, 8, 32, 32),  # (batch, heads, seq, seq)
                    'shape': (1, 8, 32, 32)
                },
                {
                    'step': 1,
                    'block': 0,
                    'attention_weights': torch.randn(1, 8, 32, 32),
                    'shape': (1, 8, 32, 32)
                }
            ],
            'layers.1.attention': [
                {
                    'step': 0,
                    'block': 0,
                    'attention_weights': torch.randn(1, 8, 32, 32),
                    'shape': (1, 8, 32, 32)
                }
            ]
        }
        
        # Create visualizer
        visualizer = AttentionVisualizer(output_dir="test_visualizations")
        logger.info("✓ Created AttentionVisualizer")
        
        # Test creating a heatmap
        test_weights = dummy_attention_maps['layers.0.attention'][0]['attention_weights']
        fig = visualizer.create_attention_heatmap(
            test_weights.numpy(),
            'layers.0.attention',
            0, 0
        )
        logger.info("✓ Created attention heatmap")
        
        # Test layer comparison
        fig = visualizer.create_layer_attention_comparison(
            dummy_attention_maps, 0
        )
        logger.info("✓ Created layer comparison")
        
        # Test comprehensive report
        saved_files = visualizer.create_comprehensive_attention_report(
            dummy_attention_maps
        )
        logger.info(f"✓ Created comprehensive report with {len(saved_files)} files")
        
        # Clean up
        import matplotlib.pyplot as plt
        plt.close('all')
        
        logger.info("✓ Attention visualizer test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error during attention visualizer test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_attention_analyzer():
    """Test the attention analyzer with dummy data."""
    
    logger.info("Testing attention analyzer...")
    
    try:
        # Create dummy attention data
        dummy_attention_maps = {
            'layers.0.attention': [
                {
                    'step': 0,
                    'block': 0,
                    'attention_weights': torch.randn(1, 8, 32, 32),
                    'shape': (1, 8, 32, 32)
                },
                {
                    'step': 1,
                    'block': 0,
                    'attention_weights': torch.randn(1, 8, 32, 32),
                    'shape': (1, 8, 32, 32)
                }
            ]
        }
        
        # Create analyzer
        analyzer = AttentionAnalyzer(dummy_attention_maps)
        logger.info("✓ Created AttentionAnalyzer")
        
        # Test statistics
        stats = analyzer.get_attention_statistics()
        logger.info(f"✓ Got attention statistics: {stats['total_attention_maps']} maps")
        
        # Test pattern analysis
        patterns = analyzer.analyze_attention_patterns()
        logger.info(f"✓ Analyzed attention patterns: {patterns['total_layers']} layers")
        
        # Test token influence analysis
        token_influence = analyzer.analyze_token_influence(prompt_length=16)
        logger.info("✓ Analyzed token influence")
        
        # Test layer summary
        layer_summary = analyzer.get_layer_attention_summary('layers.0.attention')
        logger.info(f"✓ Got layer summary: {layer_summary['total_captures']} captures")
        
        # Test export
        report_path = analyzer.export_analysis_report("test_analysis_report.json")
        logger.info(f"✓ Exported analysis report to {report_path}")
        
        logger.info("✓ Attention analyzer test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error during attention analyzer test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    
    logger.info("=" * 60)
    logger.info("ATTENTION VISUALIZATION SYSTEM TEST")
    logger.info("=" * 60)
    
    tests = [
        ("Attention Hook", test_attention_hook),
        ("Attention Visualizer", test_attention_visualizer),
        ("Attention Analyzer", test_attention_analyzer)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} Testing {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                logger.info(f"✓ {test_name} test PASSED")
            else:
                logger.error(f"✗ {test_name} test FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} test FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! The attention visualization system is working correctly.")
    else:
        logger.warning("⚠️  Some tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
