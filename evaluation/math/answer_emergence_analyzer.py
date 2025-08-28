"""
Answer Emergence Analysis for MMaDA Generation History

This module analyzes the generation history to determine when the final answer
first appears during the denoising process. This is crucial for understanding
whether the model is genuinely reasoning step-by-step or if it "knows" the
answer from early in the process.
"""

import re
import os
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
        # Strict numeric answer pattern: integers or decimals, allow optional leading sign
        self.numeric_pattern = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
    
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
    
    def is_strict_numeric(self, answer_text: str) -> bool:
        """Return True if answer is strictly numeric (int or float)."""
        return bool(self.numeric_pattern.match(answer_text.strip()))

    def is_single_digit_numeric(self, answer_text: str) -> bool:
        """Return True if answer is a single-digit integer (optionally signed)."""
        if answer_text is None:
            return False
        text = answer_text.strip()
        # Allow optional leading sign, then exactly one digit 0-9
        if text.startswith("+") or text.startswith("-"):
            text = text[1:]
        return len(text) == 1 and text.isdigit()

    def find_numeric_emergence_in_text_history(self, history_file: str, numeric_answer: str) -> Optional[Tuple[int, int, float]]:
        """Scan the text history file to find earliest step where numeric answer first appears.
        Returns (emergence_step, total_steps, completion_percentage) or None.
        """
        try:
            # Normalize target search string (strip whitespace)
            target = numeric_answer.strip()
            earliest_step: Optional[int] = None
            total_steps: Optional[int] = None

            # Patterns in the text report
            step_header_re = re.compile(r"^Step\s+(\d+)\s*\(.*\):\s*$")
            total_steps_re = re.compile(r"^Total Steps:\s*(\d+)\s*$")
            # Only consider lines that display generated text at that step
            generated_line_re = re.compile(r"^\s{2}(Generated so far:|Finalized text:)\s*(.*)$")
            # Build a boundary-aware pattern for the exact numeric target (avoid matching '8' inside '180')
            target_re = re.compile(rf"(?<![0-9]){re.escape(target)}(?![0-9])")

            with open(history_file, 'r', encoding='utf-8') as f:
                current_step: Optional[int] = None
                for line in f:
                    if total_steps is None:
                        m_tot = total_steps_re.match(line)
                        if m_tot:
                            try:
                                total_steps = int(m_tot.group(1))
                            except Exception:
                                pass
                    m = step_header_re.match(line)
                    if m:
                        try:
                            current_step = int(m.group(1))
                        except Exception:
                            current_step = None
                        continue
                    # When within a step, only consider generated/finalized lines
                    if current_step is not None:
                        mg = generated_line_re.match(line)
                        if mg:
                            text_fragment = mg.group(2)
                            if target_re.search(text_fragment):
                                if earliest_step is None:
                                    earliest_step = current_step
                                    # We can break early if we found the first occurrence, but
                                    # keep scanning to capture total_steps if not yet found
                # End for
            if earliest_step is None or total_steps is None or total_steps <= 0:
                return None
            completion_percentage = (earliest_step / total_steps) * 100.0
            return (earliest_step, total_steps, completion_percentage)
        except Exception as e:
            logger.error(f"Error scanning history file '{history_file}': {e}")
            return None

    def find_numeric_emergence_in_generated_so_far(self, history_file: str, numeric_answer: str) -> Optional[Tuple[int, int, float]]:
        """Scan the text history file to find earliest step where numeric answer first appears in 'Generated so far' text.
        This accounts for the 1-step lag where 'Generated so far' shows what was finalized up to the previous step.
        Returns (emergence_step, total_steps, completion_percentage) or None.
        """
        try:
            # Normalize target search string (strip whitespace)
            target = numeric_answer.strip()
            earliest_step: Optional[int] = None
            total_steps: Optional[int] = None

            # Patterns in the text report
            step_header_re = re.compile(r"^Step\s+(\d+)\s*\(.*\):\s*$")
            total_steps_re = re.compile(r"^Total Steps:\s*(\d+)\s*$")
            # Only consider 'Generated so far' lines (not 'Finalized text')
            generated_so_far_re = re.compile(r"^\s{2}Generated so far:\s*(.*)$")
            # Build a boundary-aware pattern for the exact numeric target (avoid matching '8' inside '180')
            target_re = re.compile(rf"(?<![0-9]){re.escape(target)}(?![0-9])")

            with open(history_file, 'r', encoding='utf-8') as f:
                current_step: Optional[int] = None
                for line in f:
                    if total_steps is None:
                        m_tot = total_steps_re.match(line)
                        if m_tot:
                            try:
                                total_steps = int(m_tot.group(1))
                            except Exception:
                                pass
                    m = step_header_re.match(line)
                    if m:
                        try:
                            current_step = int(m.group(1))
                        except Exception:
                            current_step = None
                        continue
                    # When within a step, only consider 'Generated so far' lines
                    if current_step is not None:
                        mg = generated_so_far_re.match(line)
                        if mg:
                            text_fragment = mg.group(1)
                            if target_re.search(text_fragment):
                                if earliest_step is None:
                                    earliest_step = current_step
                                    # We can break early if we found the first occurrence, but
                                    # keep scanning to capture total_steps if not yet found
                # End for
            if earliest_step is None or total_steps is None or total_steps <= 0:
                return None
            completion_percentage = (earliest_step / total_steps) * 100.0
            return (earliest_step, total_steps, completion_percentage)
        except Exception as e:
            logger.error(f"Error scanning history file '{history_file}' for 'Generated so far': {e}")
            return None

    # Existing token-based methods (kept for future)
    def tokenize_answer(self, answer_text: str) -> List[int]:
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
        if not answer_tokens:
            return False
        for i in range(len(current_tokens) - len(answer_tokens) + 1):
            if current_tokens[i:i + len(answer_tokens)] == answer_tokens:
                return True
        return False
    
    def find_answer_emergence(self, history: Dict[str, Any], final_answer: str) -> Optional[Dict[str, Any]]:
        if not history or 'states' not in history:
            logger.error("Invalid history format: missing 'states'")
            return None
        answer_tokens = self.tokenize_answer(final_answer)
        if not answer_tokens:
            logger.error("Failed to tokenize answer")
            return None
        logger.info(f"Searching for {len(answer_tokens)} answer tokens across {len(history['states'])} states")
        for step_idx, state in enumerate(history['states']):
            try:
                if isinstance(state, (list, tuple)):
                    current_tokens = state[0].cpu().tolist() if hasattr(state[0], 'cpu') else state[0]
                else:
                    current_tokens = state.cpu().tolist() if hasattr(state, 'cpu') else state
                if self.contains_answer_tokens(current_tokens, answer_tokens):
                    step_info = history.get('step_info', [])
                    step_details = step_info[step_idx] if step_idx < len(step_info) else {}
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
        try:
            if isinstance(history_data, dict):
                history = history_data
            else:
                import pickle
                with open(history_data, 'rb') as f:
                    history = pickle.load(f)
            final_answer = self.extract_final_answer(completion_text)
            if not final_answer:
                return None
            return self.find_answer_emergence(history, final_answer)
        except Exception as e:
            logger.error(f"Error analyzing generation history: {e}")
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
            # Extract final answer and optionally skip single-digit numeric answers to avoid noisy early matches
            final_answer = analyzer.extract_final_answer(completion)
            skip_single_digit = os.environ.get("ANSWER_EMERGENCE_SKIP_SINGLE_DIGIT", "true").lower() in {"1","true","yes","on"}
            if skip_single_digit and final_answer and analyzer.is_strict_numeric(final_answer) and analyzer.is_single_digit_numeric(final_answer):
                return {
                    'value': None,
                    'metadata': {
                        'answer_emergence_analysis': 'skipped_single_digit',
                        'final_answer': final_answer
                    },
                    'explanation': 'Skipped single-digit numeric answer due to noisy early digits'
                }
            
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

