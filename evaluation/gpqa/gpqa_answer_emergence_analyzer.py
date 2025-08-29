"""
GPQA Answer Emergence Analysis for MMaDA Generation History

This module analyzes the generation history to determine when the final multiple choice answer
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


class GPQAAnswerEmergenceAnalyzer:
    """Analyzer for detecting when multiple choice answers first appear during generation."""
    
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        # Use LETTER pattern for multiple choice answers (A, B, C, D)
        self.answer_pattern = re.compile(AnswerPattern.LETTER, re.MULTILINE)
        # Multiple choice answer pattern: single letter A, B, C, or D
        self.multiple_choice_pattern = re.compile(r"^[ABCD]$")
    
    def extract_final_answer(self, completion_text: str) -> Optional[str]:
        """Extract the final multiple choice answer from completion text."""
        if not completion_text:
            return None
        
        # First try to find answer in the standard format (e.g., "The answer is A")
        matches = self.answer_pattern.findall(completion_text)
        if matches:
            # Get the last match (in case there are multiple answer lines)
            final_answer = matches[-1].strip()
            logger.info(f"Extracted final answer: '{final_answer}'")
            return final_answer
        
        # If no standard format found, check if the completion ends with just a letter
        # This catches cases where the model outputs just "A", "B", "C", or "D"
        stripped_completion = completion_text.strip()
        if self.multiple_choice_pattern.match(stripped_completion):
            logger.info(f"Extracted final answer from standalone letter: '{stripped_completion}'")
            return stripped_completion
        
        # Also check for common variations like "Answer: A" or "Final answer: B"
        answer_variations = re.compile(r"(?:answer|final answer|correct answer|choice)\s*:?\s*([ABCD])", re.IGNORECASE)
        variation_matches = answer_variations.findall(completion_text)
        if variation_matches:
            final_answer = variation_matches[-1].strip()
            logger.info(f"Extracted final answer from variation: '{final_answer}'")
            return final_answer
        
        logger.warning(f"No multiple choice answer found in completion: {completion_text[:100]}...")
        return None
    
    def is_multiple_choice_answer(self, answer_text: str) -> bool:
        """Return True if answer is a valid multiple choice answer (A, B, C, or D)."""
        return bool(self.multiple_choice_pattern.match(answer_text.strip()))

    def find_answer_emergence_in_text_history(self, history_file: str, answer: str) -> Optional[Tuple[int, int, float]]:
        """Scan the text history file to find earliest step where answer first appears.
        Returns (emergence_step, total_steps, completion_percentage) or None.
        """
        try:
            # Normalize target search string (strip whitespace)
            target = answer.strip()
            earliest_step: Optional[int] = None
            total_steps: Optional[int] = None

            # Patterns in the text report
            step_header_re = re.compile(r"^Step\s+(\d+)\s*\(.*\):\s*$")
            total_steps_re = re.compile(r"^Total Steps:\s*(\d+)\s*$")
            # Only consider lines that display generated text at that step
            generated_line_re = re.compile(r"^\s{2}(?:Generated so far:|Finalized text:)\s*(.*)$")
            # Build a boundary-aware pattern for the exact answer (avoid matching 'A' inside words)
            target_re = re.compile(rf"(?<![A-Za-z]){re.escape(target)}(?![A-Za-z])")

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
            logger.error(f"Error scanning history file '{history_file}': {e}")
            return None

    def find_answer_emergence_in_generated_so_far(self, history_file: str, answer: str) -> Optional[Tuple[int, int, float]]:
        """Scan the text history file to find earliest step where answer first appears in 'Generated so far' text.
        This accounts for the 1-step lag where 'Generated so far' shows what was finalized up to the previous step.
        Returns (emergence_step, total_steps, completion_percentage) or None.
        """
        try:
            # Normalize target search string (strip whitespace)
            target = answer.strip()
            earliest_step: Optional[int] = None
            total_steps: Optional[int] = None

            # Patterns in the text report
            step_header_re = re.compile(r"^Step\s+(\d+)\s*\(.*\):\s*$")
            total_steps_re = re.compile(r"^Total Steps:\s*(\d+)\s*$")
            # Only consider 'Generated so far' lines (not 'Finalized text')
            generated_so_far_re = re.compile(r"^\s{2}Generated so far:\s*(.*)$")
            # Build a boundary-aware pattern for the exact answer (avoid matching 'A' inside words)
            target_re = re.compile(rf"(?<![A-Za-z]){re.escape(target)}(?![A-Za-z])")

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
        
        # Tokenize the answer
        answer_tokens = self.tokenize_answer(final_answer)
        if not answer_tokens:
            logger.error(f"Could not tokenize answer: {final_answer}")
            return None
        
        # Scan through states to find when answer first appears
        states = history['states']
        for i, state in enumerate(states):
            current_tokens = state.get('tokens', [])
            if self.contains_answer_tokens(current_tokens, answer_tokens):
                return {
                    'emergence_step': i,
                    'total_steps': len(states),
                    'completion_percentage': (i / len(states)) * 100.0,
                    'method': 'token_scan'
                }
        
        logger.warning(f"Answer '{final_answer}' not found in any generation state")
        return None
