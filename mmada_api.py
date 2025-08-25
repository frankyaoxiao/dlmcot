#!/usr/bin/env python3
"""
MMaDA API - Simple interface for MMaDA text generation
Using the original repository approach
"""

import argparse
import sys
from mmada_inference import generate_text

def main():
    parser = argparse.ArgumentParser(description="Generate text using MMaDA")
    parser.add_argument("prompt", type=str, help="Input prompt for generation")
    parser.add_argument("--gen_length", type=int, default=128, help="Generation length (default: 128)")
    parser.add_argument("--steps", type=int, default=128, help="Number of diffusion steps (default: 128)")
    parser.add_argument("--block_length", type=int, default=32, help="Block length (default: 32)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling (default: 1.0)")
    parser.add_argument("--cfg_scale", type=float, default=0.0, help="CFG scale (default: 0.0)")
    parser.add_argument("--remasking", type=str, default="low_confidence", 
                       choices=["low_confidence", "random"], help="Remasking strategy (default: low_confidence)")
    parser.add_argument("--save_history", action="store_true", help="Save generation history")
    parser.add_argument("--use_chat_template", type=bool, default=True, help="Use chat template for conversational prompts (default: True)")
    parser.add_argument("--enable_cot", action="store_true", help="Enable Chain-of-Thought reasoning with <think> tags")
    
    args = parser.parse_args()
    
    try:
        if args.save_history:
            generated_text, history, tokenizer = generate_text(
                args.prompt,
                gen_length=args.gen_length,
                steps=args.steps,
                block_length=args.block_length,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
                remasking=args.remasking,
                save_history=True,
                use_chat_template=args.use_chat_template,
                enable_cot=args.enable_cot
            )
            print("Generated text with history:")
            print(generated_text)
            print(f"\nHistory saved with {len(history['states'])} states")
        else:
            generated_text = generate_text(
                args.prompt,
                gen_length=args.gen_length,
                steps=args.steps,
                block_length=args.block_length,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
                remasking=args.remasking,
                save_history=False,
                use_chat_template=args.use_chat_template,
                enable_cot=args.enable_cot
            )
            print("Generated text:")
            print(generated_text)
            
    except Exception as e:
        print(f"Error during generation: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
