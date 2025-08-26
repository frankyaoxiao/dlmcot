#!/usr/bin/env python3
"""
Test script for MMaDA integration with Inspect AI
"""

import asyncio
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from evaluation.model_api.mmada_model_api import MMadaModelAPI
from inspect_ai.model import GenerateConfig, ChatMessageUser


async def test_mmada_integration():
    """Test the MMaDA integration with Inspect AI."""
    
    print("Testing MMaDA integration with Inspect AI...")
    
    # Create MMaDA model API instance
    model_api = MMadaModelAPI(
        model_name="mmada/8b-mixcot",
        config=GenerateConfig(
            temperature=1.0,
            max_tokens=256,
            enable_cot=True
        )
    )
    
    # Test messages
    messages = [
        ChatMessageUser(content="What is 12 + 57? Please show your reasoning.")
    ]
    
    print("Generating response...")
    
    try:
        # Generate response
        response = await model_api.generate(
            input=messages,
            tools=[],
            tool_choice=None,
            config=GenerateConfig(
                temperature=1.0,
                max_tokens=256,
                enable_cot=True
            )
        )
        
        print("✅ MMaDA integration successful!")
        print(f"Response: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"❌ MMaDA integration failed: {e}")
        import traceback
        traceback.print_exc()


async def test_simple_generation():
    """Test simple text generation without Inspect AI."""
    
    print("\nTesting simple MMaDA generation...")
    
    try:
        from mmada_inference import generate_text
        
        response = generate_text(
            prompt="What is 12 + 57?",
            gen_length=128,
            temperature=1.0,
            enable_cot=True
        )
        
        print("✅ Simple generation successful!")
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"❌ Simple generation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_simple_generation())
    asyncio.run(test_mmada_integration())
