#!/usr/bin/env python3
"""
Test script for MMaDA original approach (loading from Hugging Face)
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported."""
    try:
        import torch
        import transformers
        import numpy as np
        from models import MMadaModelLM
        from mmada_inference import generate_text
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_basic_generation():
    """Test basic text generation."""
    try:
        from mmada_inference import generate_text
        
        prompt = "Hello, how are you?"
        print(f"Testing generation with prompt: '{prompt}'")
        
        generated_text = generate_text(
            prompt,
            gen_length=32,
            steps=32,
            block_length=32,
            temperature=0.0,
            cfg_scale=0.0,
            remasking='low_confidence',
            save_history=False,
            use_chat_template=False
        )
        
        print(f"✓ Generation successful!")
        print(f"Generated text: {generated_text}")
        return True
        
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chat_generation():
    """Test chat-style generation."""
    try:
        from mmada_inference import generate_text
        
        prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"
        print(f"Testing chat generation with prompt: '{prompt}'")
        
        generated_text = generate_text(
            prompt,
            gen_length=128,
            steps=128,
            block_length=128,
            temperature=1.0,
            cfg_scale=0.0,
            remasking='low_confidence',
            save_history=False,
            use_chat_template=True
        )
        
        print(f"✓ Chat generation successful!")
        print(f"Generated text: {generated_text}")
        return True
        
    except Exception as e:
        print(f"✗ Chat generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Testing MMaDA setup...")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("Import test failed. Please check your environment.")
        return False
    
    # Test basic generation
    if not test_basic_generation():
        print("Basic generation test failed.")
        return False
    
    # Test chat generation
    if not test_chat_generation():
        print("Chat generation test failed.")
        return False
    
    print("=" * 50)
    print("✓ All tests passed! MMaDA is working correctly.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
