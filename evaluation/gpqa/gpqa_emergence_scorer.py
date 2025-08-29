"""
GPQA Answer Emergence Scorer for MMaDA

This module provides a scorer that analyzes when multiple choice answers first appear
during the generation process, similar to the MATH emergence scorer.
"""

import os
import logging
from typing import Any

from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState

from evaluation.gpqa.gpqa_answer_emergence_analyzer import GPQAAnswerEmergenceAnalyzer

logger = logging.getLogger(__name__)


@scorer(metrics=[])  # No metrics since this is analysis-only
def gpqa_answer_emergence_scorer(use_generated_so_far: bool = False) -> Scorer:
    """
    Create a scorer that analyzes when multiple choice answers first appear during generation.
    
    Args:
        use_generated_so_far: If True, find when answer first appears in "Generated so far" text
                             (accounting for 1-step lag). If False, use current method.
    """
    
    async def score(state: TaskState, target: Target) -> Score:
        """Analyze answer emergence timing from generation history."""
        try:
            # Get the completion text
            completion = state.output.completion
            if not completion:
                return Score(
                    value=0.0,
                    metadata={'answer_emergence_analysis': 'no_completion'}
                )
            
            # Check if we have generation history metadata
            # The metadata is stored in state.output.metadata, not model_output.metadata
            metadata = getattr(state.output, 'metadata', {})
            if not metadata or 'generation_history' not in metadata:
                return Score(
                    value=0.0,
                    metadata={'answer_emergence_analysis': 'no_history'}
                )
            
            # Extract generation history information
            history_info = metadata['generation_history']
            total_steps = history_info.get('total_steps', 0)
            total_states = history_info.get('total_states', 0)
            history_file = history_info.get('history_file', None)
            
            # Create analyzer and extract final answer
            analyzer = GPQAAnswerEmergenceAnalyzer()
            final_answer = analyzer.extract_final_answer(completion)
            
            print(f"DEBUG: Completion text: '{completion}'")
            print(f"DEBUG: Extracted answer: '{final_answer}'")
            print(f"DEBUG: Using 'Generated so far' method: {use_generated_so_far}")
            
            if not final_answer:
                return Score(
                    value=-1.0,
                    metadata={'answer_emergence_analysis': 'no_answer_extracted'}
                )
            
            # Check if it's a valid multiple choice answer
            if analyzer.is_multiple_choice_answer(final_answer) and history_file and os.path.exists(history_file):
                if use_generated_so_far:
                    # Use new method: find answer in "Generated so far" text
                    answer_result = analyzer.find_answer_emergence_in_generated_so_far(history_file, final_answer)
                    method_used = 'generated_so_far_scan'

                    # Fallback: if not found in Generated so far (due to 1-step lag at final step),
                    # check if it appears in the last step's Finalized text and attribute emergence to last step.
                    if answer_result is None:
                        tail_result = analyzer.find_answer_emergence_in_text_history(history_file, final_answer)
                        if tail_result is not None:
                            es, ts, _ = tail_result
                            # Only accept fallback when the first appearance is exactly at the last step
                            if ts and es == ts - 1:
                                emergence_step, total_steps_detected = es, ts
                                completion_percentage = (emergence_step / total_steps_detected) * 100.0
                                return Score(
                                    value=completion_percentage / 100.0,
                                    metadata={
                                        'answer_emergence_analysis': 'generated_so_far_fallback_last_step',
                                        'final_answer': final_answer,
                                        'emergence_step': emergence_step,
                                        'total_steps': total_steps_detected,
                                        'completion_percentage': completion_percentage,
                                        'history_file': history_file,
                                        'method_used': method_used,
                                        'use_generated_so_far': use_generated_so_far
                                    }
                                )
                else:
                    # Use original method: find answer in any generated text
                    answer_result = analyzer.find_answer_emergence_in_text_history(history_file, final_answer)
                    method_used = 'text_history_scan'
                
                if answer_result is not None:
                    emergence_step, total_steps_detected, completion_percentage = answer_result
                    return Score(
                        value=completion_percentage / 100.0,
                        metadata={
                            'answer_emergence_analysis': f'{method_used}_success',
                            'final_answer': final_answer,
                            'emergence_step': emergence_step,
                            'total_steps': total_steps_detected,
                            'completion_percentage': completion_percentage,
                            'history_file': history_file,
                            'method_used': method_used,
                            'use_generated_so_far': use_generated_so_far
                        }
                    )
                else:
                    return Score(
                        value=0.0,
                        metadata={
                            'answer_emergence_analysis': f'{method_used}_not_found',
                            'final_answer': final_answer,
                            'total_steps': total_steps,
                            'total_states': total_states,
                            'history_file': history_file,
                            'method_used': method_used,
                            'use_generated_so_far': use_generated_so_far
                        }
                    )
            
            # If not a valid multiple choice answer or no history file, return basic metadata and skip
            return Score(
                value=0.0,
                metadata={
                    'answer_emergence_analysis': 'skipped_invalid_answer_or_missing_history',
                    'final_answer': final_answer,
                    'is_multiple_choice_answer': analyzer.is_multiple_choice_answer(final_answer),
                    'total_steps': total_steps,
                    'total_states': total_states,
                    'history_file': history_file,
                    'method_used': 'none',
                    'use_generated_so_far': use_generated_so_far
                }
            )
            
        except Exception as e:
            return Score(
                value=0.0,
                metadata={'answer_emergence_analysis': f'error: {str(e)}'}
            )
    
    return score
