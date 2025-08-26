#!/usr/bin/env python3
"""
Direct test of GPQA with MMaDA - bypassing inspect to test our integration
"""

import asyncio
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from evaluation.model_api.mmada_model_api import MMadaModelAPI
from inspect_ai.model import GenerateConfig, ChatMessageUser


async def test_gpqa_direct():
    """Test GPQA questions directly with MMaDA."""
    
    print("Testing GPQA questions directly with MMaDA...")
    
    # Create MMaDA model API instance
    model_api = MMadaModelAPI(
        model_name="mmada/8b-mixcot",
        config=GenerateConfig(
            temperature=1.0,
            max_tokens=256,
            enable_cot=False  # No CoT for basic test
        )
    )
    
    # Sample GPQA questions
    gpqa_questions = [
        "Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec, respectively. We want to clearly distinguish these two energy levels. Which one of the following options could be their energy difference so that they can be clearly resolved?\n\nA) 10^-4 eV\nB) 10^-11 eV\nC) 10^-8 eV\nD) 10^-9 eV",
        
        "What is the capital of France?\n\nA) London\nB) Paris\nC) Berlin\nD) Madrid"
    ]
    
    for i, question in enumerate(gpqa_questions):
        print(f"\n--- Question {i+1} ---")
        print(f"Question: {question[:100]}...")
        
        # Format as multiple choice prompt
        prompt = f"""Answer the following multiple choice question. 
The entire content of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of A,B,C,D.

{question}"""
        
        messages = [ChatMessageUser(content=prompt)]
        
        try:
            # Generate response
            response = await model_api.generate(
                input=messages,
                tools=[],
                tool_choice=None,
                config=GenerateConfig(
                    temperature=1.0,
                    max_tokens=256,
                    enable_cot=False
                )
            )
            
            print(f"Response: {response.choices[0].message.content}")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_gpqa_direct())
