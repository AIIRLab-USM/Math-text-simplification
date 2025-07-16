"""
parser.py

This script extracts a sample of math-related questions from a StackExchange-style XML dump.
It cleans the HTML content, identifies LaTeX-style math expressions, and saves the cleaned
data in JSONL format for downstream processing (e.g., simplification, annotation).

Usage:
    python parser.py Posts.xml 10
"""

import sys
import os
import json
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

def extract_and_clean_math(html):
    """
    Extracts LaTeX-style math from <span class="math-container"> tags in the HTML.
    Replaces them with [MATH]...[/MATH] tags to preserve during LLM processing.

    Args:
        html (str): Raw HTML content.

    Returns:
        tuple: (cleaned text without HTML, list of extracted LaTeX expressions)
    """
    soup = BeautifulSoup(html, "html.parser")
    math_latex = []

    for span in soup.find_all("span", {"class": "math-container"}):
        latex = span.get_text()
        math_latex.append(latex)
        span.replace_with(f"[MATH]{latex}[/MATH]")

    return soup.get_text(), math_latex

def extract_questions(xml_path, max_questions=5):
    """
    Parses XML to extract a limited number of questions (PostTypeId=1).
    Cleans and normalizes each question’s text and metadata.

    Args:
        xml_path (str): Path to Posts.xml file.
        max_questions (int): Maximum number of questions to extract.

    Returns:
        list: Extracted and cleaned question dictionaries.
    """
    questions = []
    context = ET.iterparse(xml_path, events=("end",))

    for _, elem in context:
        if elem.tag != "row":
            continue

        attrib = elem.attrib

        if attrib.get("PostTypeId") != "1":  # Skip non-question posts
            elem.clear()
            continue

        post_id = int(attrib.get("Id", -1))
        score = int(attrib.get("Score", 0))
        view_count = int(attrib.get("ViewCount", 0))
        body = attrib.get("Body", "")
        raw_title = attrib.get("Title", "")
        tags_raw = attrib.get("Tags", "")

        #Clean title and extract LaTeX
        clean_title, title_latex = extract_and_clean_math(raw_title)
        clean_title = clean_title.replace("\\", "\\\\").replace("\r", "").strip()

        # Extract math and clean text
        clean_body, math_expressions = extract_and_clean_math(body)
        clean_body = clean_body.replace("\\", "\\\\").replace("\r", "").strip()

        # Parse tags into list
        tags = tags_raw.replace("><", ",").replace("<", "").replace(">", "").split(",") if tags_raw else []

        question = {
            "post_id": post_id,
            "title": clean_title,
            "latex_title": title_latex,
            "tags": tags,
            "body": clean_body,
            "latex": math_expressions,
            "score": score,
            "view_count": view_count
        }

        questions.append(question)
        print(f"[{len(questions)}] Sampled: {clean_title[:60]}...")

        elem.clear()
        if len(questions) >= max_questions:
            break

    return questions

def save_questions_to_jsonl(questions, output_file="cleaned_data.jsonl"):
    """
    Saves a list of question dictionaries to a JSONL file.

    Args:
        questions (list): List of question dicts to save.
        output_file (str): Path to output JSONL file.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(questions)} questions to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python parser.py Posts.xml 10")
        sys.exit(1)

    xml_file = sys.argv[1]
    num_samples = int(sys.argv[2])

    if not os.path.isfile(xml_file):
        print(f"File not found: {xml_file}")
        sys.exit(1)

    print(f"Reading from: {xml_file}")
    print(f"Extracting {num_samples} questions...\n")

    questions = extract_questions(xml_file, max_questions=num_samples)
    save_questions_to_jsonl(questions)
