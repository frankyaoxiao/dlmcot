"""
Basic GPQA evaluation with MMaDA
Simple test to ensure GPQA works with MMaDA before adding hints
"""

from typing import Any
import sys
import os

# Add the parent directory to the path to import our MMaDA inference module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, csv_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice
from inspect_ai.model import GenerateConfig

# Import our MMaDA model API
from evaluation.model_api.mmada_model_api import mmada_provider

# Default epochs to run eval for
DEFAULT_EPOCHS = 1  # Start with 1 for testing


@task
def gpqa_basic() -> Task:
    """
    Basic GPQA evaluation with MMaDA - no hints, just standard multiple choice
    """
    
    # Create the base dataset
    dataset = csv_dataset(
        csv_file="https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv",
        sample_fields=record_to_sample,
        shuffle_choices=True,
    )
    
    # Create solver - standard multiple choice without CoT
    solver = [
        multiple_choice(
            cot=False,
            prompt_template="""Answer the following multiple choice question. 
The entire content of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of A,B,C,D.

{prompt}"""
        )
    ]
    
    return Task(
        dataset=dataset,
        solver=solver,
        scorer=[choice()],
        epochs=DEFAULT_EPOCHS,
        config=GenerateConfig(
            temperature=1.0,
            max_tokens=256,
            enable_cot=False  # No CoT for basic test
        )
    )


def record_to_sample(record: dict[str, Any]) -> Sample:
    """Convert dataset record to sample."""
    
    return Sample(
        input=record["Question"],
        choices=[
            str(record["Correct Answer"]),
            str(record["Incorrect Answer 1"]),
            str(record["Incorrect Answer 2"]),
            str(record["Incorrect Answer 3"]),
        ],
        target="A",
        id=record["Record ID"],
    )
