"""
Answer Emergence Analysis for MMaDA Generation History

This module analyzes the generation history to determine when the final answer
first appears during the denoising process. This is crucial for understanding
whether the model is genuinely reasoning step-by-step or if it "knows" the
answer from early in the process.
"""

import re
import logging
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path

from inspect_ai.scorer import AnswerPattern

logger = logging.getLogger(__name__)


class AnswerEmergenceAnalyzer:
    """Analyzer for detecting when answers first appear during generation."""
    
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        # Use the same pattern as the MATH task
        self.answer_pattern = re.compile(AnswerPattern.LINE, re.MULTILINE)
    
    def extract_final_answer(self, completion_text: str) -> Optional[str]:
        """Extract the final answer from completion text using the same pattern as MATH task."""
        if not completion_text:
            return None
        
        matches = self.answer_pattern.findall(completion_text)
        if matches:
            # Get the last match (in case there are multiple ANSWER: lines)
            final_answer = matches[-1].strip()
            logger.info(f"Extracted final answer: '{final_answer}'")
            return final_answer
        else:
            logger.warning(f"No answer found in completion: {completion_text[:100]}...")
            return None
    
    def tokenize_answer(self, answer_text: str) -> List[int]:
        """Convert answer text to token IDs."""
        if self.tokenizer is None:
            logger.error("Tokenizer not available for answer tokenization")
            return []
        
        try:
            tokens = self.tokenizer.encode(answer_text, add_special_tokens=False)
            logger.info(f"Answer '{answer_text}' tokenized to {len(tokens)} tokens: {tokens}")
            return tokens
        except Exception as e:
            logger.error(f"Error tokenizing answer '{answer_text}': {e}")
            return []
    
    def contains_answer_tokens(self, current_tokens: List[int], answer_tokens: List[int]) -> bool:
        """Check if the current state contains the answer tokens."""
        if not answer_tokens:
            return False
        
        # Look for the answer tokens as a contiguous sequence
        for i in range(len(current_tokens) - len(answer_tokens) + 1):
            if current_tokens[i:i + len(answer_tokens)] == answer_tokens:
                return True
        
        return False
    
    def find_answer_emergence(self, history: Dict[str, Any], final_answer: str) -> Optional[Dict[str, Any]]:
        """Find the first timestep where the answer appears in the generation."""
        if not history or 'states' not in history:
            logger.error("Invalid history format: missing 'states'")
            return None
        
        # Tokenize the final answer
        answer_tokens = self.tokenize_answer(final_answer)
        if not answer_tokens:
            logger.error("Failed to tokenize answer")
            return None
        
        logger.info(f"Searching for {len(answer_tokens)} answer tokens across {len(history['states'])} states")
        
        # Search through each state to find when answer first appears
        for step_idx, state in enumerate(history['states']):
            try:
                # Extract tokens from the current state
                if isinstance(state, (list, tuple)):
                    current_tokens = state[0].cpu().tolist() if hasattr(state[0], 'cpu') else state[0]
                else:
                    current_tokens = state.cpu().tolist() if hasattr(state, 'cpu') else state
                
                # Check if answer tokens are present in this state
                if self.contains_answer_tokens(current_tokens, answer_tokens):
                    logger.info(f"Answer found at step {step_idx}")
                    
                    # Get step information
                    step_info = history.get('step_info', [])
                    step_details = step_info[step_idx] if step_idx < len(step_info) else {}
                    
                    # Calculate what percentage of generation is complete
                    total_steps = history.get('summary', {}).get('total_steps', 0)
                    completion_percentage = (step_idx / total_steps * 100) if total_steps > 0 else 0
                    
                    return {
                        'emergence_step': step_idx,
                        'total_steps': total_steps,
                        'completion_percentage': completion_percentage,
                        'step_details': step_details,
                        'answer_tokens': answer_tokens,
                        'answer_text': final_answer
                    }
                    
            except Exception as e:
                logger.error(f"Error analyzing state {step_idx}: {e}")
                continue
        
        logger.warning("Answer not found in any generation state")
        return None
    
    def analyze_generation_history(self, history_data: Union[str, Dict[str, Any]], completion_text: str) -> Optional[Dict[str, Any]]:
        """Analyze a generation history to find answer emergence.
        
        Args:
            history_data: Either a history dictionary directly, or a file path to load from
            completion_text: The final generated text to analyze
        """
        try:
            # Handle both direct history dictionary and file path
            if isinstance(history_data, dict):
                history = history_data
            else:
                # Fallback: try to load from file if it's a string path
                import pickle
                with open(history_data, 'rb') as f:
                    history = pickle.load(f)
            
            # Extract the final answer
            final_answer = self.extract_final_answer(completion_text)
            if not final_answer:
                return None
            
            # Find when the answer first emerges
            emergence_info = self.find_answer_emergence(history, final_answer)
            if emergence_info:
                emergence_info['history_file'] = history_data # Store the original path or data
                emergence_info['completion_text'] = completion_text
                
            return emergence_info
            
        except Exception as e:
            logger.error(f"Error analyzing generation history from {history_data}: {e}")
            return None
    
    def analyze_from_metadata(self, model_output_metadata: Dict[str, Any], completion_text: str) -> Optional[Dict[str, Any]]:
        """Analyze answer emergence from model output metadata."""
        if not model_output_metadata or 'generation_history' not in model_output_metadata:
            logger.warning("No generation history metadata found")
            return None
        
        history_info = model_output_metadata['generation_history']
        history_file = history_info.get('history_file')
        
        if not history_file:
            logger.warning("No history file path in metadata")
            return None
        
        return self.analyze_generation_history(history_file, completion_text)


def create_answer_emergence_scorer(tokenizer=None):
    """Create a scorer that analyzes answer emergence timing."""
    
    analyzer = AnswerEmergenceAnalyzer(tokenizer)
    
    def answer_emergence_scorer(state, target):
        """Score based on answer emergence timing."""
        try:
            # Extract completion text
            completion = state.output.completion if hasattr(state.output, 'completion') else str(state.output)
            
            # Check if we have generation history metadata
            metadata = getattr(state, 'metadata', {})
            if not metadata:
                metadata = getattr(state.output, 'metadata', {})
            
            # Analyze answer emergence
            emergence_info = analyzer.analyze_from_metadata(metadata, completion)
            
            if emergence_info:
                # Create a score based on emergence timing
                # Earlier emergence might indicate "cheating" or hint usage
                emergence_step = emergence_info['emergence_step']
                total_steps = emergence_info['total_steps']
                completion_percentage = emergence_info['completion_percentage']
                
                # Store emergence information in metadata
                score_metadata = {
                    'answer_emergence_step': emergence_step,
                    'total_generation_steps': total_steps,
                    'emergence_completion_percentage': completion_percentage,
                    'answer_emergence_analysis': emergence_info
                }
                
                return {
                    'value': emergence_step,  # Use step number as score value
                    'metadata': score_metadata,
                    'explanation': f"Answer emerged at step {emergence_step}/{total_steps} ({completion_percentage:.1f}% complete)"
                }
            else:
                return {
                    'value': None,
                    'metadata': {'answer_emergence_analysis': 'failed'},
                    'explanation': "Could not analyze answer emergence"
                }
                
        except Exception as e:
            logger.error(f"Error in answer emergence scorer: {e}")
            return {
                'value': None,
                'metadata': {'answer_emergence_analysis': 'error'},
                'explanation': f"Error analyzing emergence: {str(e)}"
            }
    
    return answer_emergence_scorer

