"""
annotate.py

This script takes cleaned math question data (in JSONL format), identifies mathematical terms,
computes their difficulty levels, simplifies the text using a language model, and outputs
an annotated version with term definitions and difficulty ratings.

Dependencies:
- definitions.py
- llm_definitions.py
- llm_simplifier.py
- models.py

Usage:
    python annotate.py cleaned_data.jsonl annotated_output.jsonl
"""

import json
import sys
import os
import re
from models import pipeline_llama
from definitions import easy_term, medium_term, hard_term
from llm_definitions import get_definitions
from llm_simplifier import simplify_math_text


def format_latex_math(text):
    """
    Replace custom math tags [MATH]...[/MATH] with LaTeX-style inline math: $...$
    """
    return re.sub(r"\[MATH\](.*?)\[/MATH\]", r"$\1$", text)


def load_cleaned_data(filepath):
    """
    Load line-separated JSON entries from a file.

    Args:
        filepath (str): Path to the cleaned JSONL file.
    Returns:
        List[dict]: List of JSON-decoded question entries.
    """
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def filter_terms_in_passage(terms_dict, passage):
    """
    Filter out terms that do not actually appear in the original passage.

    Args:
        terms_dict (dict): Detected terms and their difficulty.
        passage (str): Original question text.
    Returns:
        dict: Filtered dictionary of valid terms.
    """
    return {
        term: level for term, level in terms_dict.items()
        if term.lower() in passage.lower()
    }


def compute_overall_difficulty(terms):
    """
    Compute the overall difficulty level of a passage based on a weighted average of term difficulties.

    Args:
        terms (dict): Dictionary of terms and their difficulty levels.
    Returns:
        str: One of "easy", "medium", or "hard".
    """
    if not terms:
        return "easy"

    score_map = {"easy": 1, "medium": 2, "hard": 3}
    reverse_map = {1: "easy", 2: "medium", 3: "hard"}

    term_scores = [score_map[difficulty] for difficulty in terms.values()]
    avg_score = round(sum(term_scores) / len(term_scores))
    avg_score = max(1, min(avg_score, 3))  # Clamp between 1 and 3

    return reverse_map[avg_score]


def annotate_data(cleaned_data, output_path):
    """
    Annotate cleaned math question data with simplified text, detected terms, and difficulty.

    Args:
        cleaned_data (list): List of preprocessed math questions.
        output_path (str): Destination path for annotated output.
    """
    annotated = []

    for i, sample in enumerate(cleaned_data):
        print(f"[{i+1}/{len(cleaned_data)}] Processing post_id: {sample['post_id']}...")

        raw = format_latex_math(sample["body"])

        try:
            terms = get_definitions(raw)
            terms = filter_terms_in_passage(terms, raw)
        except Exception as e:
            print(f"Error getting definitions: {e}")
            terms = {}

        try:
            simplified = simplify_math_text(raw, terms)
        except Exception as e:
            print(f"Error simplifying text: {e}")
            simplified = raw

        difficulty = compute_overall_difficulty(terms)

        annotated_sample = {
            "post_id": sample["post_id"],
            "title": sample.get("title", ""),
            "latex_title": sample.get("latex_title", ""),
            "tags": sample.get("tags", []),
            "raw_passage": raw,
            "terms": terms,
            "simplified_passage": simplified,
            "difficulty": difficulty
        }

        json_line = json.dumps(annotated_sample, ensure_ascii=False)
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json_line + "\n")

        annotated.append(annotated_sample)

    print(f"\nDone. Annotated {len(annotated)} samples.")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python annotate.py cleaned_data.jsonl annotated_output.jsonl")
        sys.exit(1)

    cleaned_file = sys.argv[1]
    output_file = sys.argv[2]

    if not os.path.exists(cleaned_file):
        print(f"File not found: {cleaned_file}")
        sys.exit(1)

    data = load_cleaned_data(cleaned_file)
    annotate_data(data, output_file)
