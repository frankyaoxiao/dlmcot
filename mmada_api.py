#!/usr/bin/env python3
"""
MMaDA API - Simple interface for MMaDA text generation
Using the original repository approach
"""

import argparse
import sys
from mmada_inference import generate_text

def main():
    parser = argparse.ArgumentParser(description="Generate text using MMaDA model")
    parser.add_argument("prompt", type=str, help="Input prompt for text generation")
    parser.add_argument("--gen_length", type=int, default=128, help="Generation length (default: 128)")
    parser.add_argument("--steps", type=int, default=128, help="Number of diffusion steps (default: 128)")
    parser.add_argument("--block_length", type=int, default=32, help="Block length for semi-autoregressive generation (default: 32)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling (default: 1.0)")
    parser.add_argument("--cfg_scale", type=float, default=0.0, help="CFG scale for guidance (default: 0.0)")
    parser.add_argument("--remasking", type=str, default="low_confidence", choices=["low_confidence", "random"], 
                       help="Remasking strategy (default: low_confidence)")
    parser.add_argument("--enable_cot", action="store_true", help="Enable Chain-of-Thought reasoning")
    parser.add_argument("--save_history", action="store_true", help="Save detailed generation history to file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic generation (default: None)")
    
    args = parser.parse_args()
    
    try:
        if args.save_history:
            generated_text, history, tokenizer, history_file = generate_text(
                args.prompt, 
                gen_length=args.gen_length, 
                steps=args.steps, 
                block_length=args.block_length, 
                temperature=args.temperature, 
                cfg_scale=args.cfg_scale, 
                remasking=args.remasking, 
                save_history=True, 
                enable_cot=args.enable_cot,
                seed=args.seed
            )
            print(f"Generated text with history:\n{generated_text}")
            print(f"\nHistory saved with {len(history['states'])} states")
            print(f"History file saved to: {history_file}")
            
            # Print summary statistics
            summary = history['summary']
            print(f"Total steps: {summary['total_steps']}")
            print(f"Finalized tokens: {summary['finalized_tokens_count']}")
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
                enable_cot=args.enable_cot,
                seed=args.seed
            )
            print(f"Generated text:\n{generated_text}")
            
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
