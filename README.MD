# Math Text Simplifier

This project uses the **Meta-LLaMA 3 (8B Instruct)** model to simplify complex scientific or mathematical sentences into more beginner-friendly language.

## Features

- Uses **Meta-LLaMA/Meta-LLaMA-3-8B-Instruct** via Hugging Face's `transformers` pipeline
- Simplifies math/science text using natural language prompts

## Requirements
The following Python packages (installed via `requirements.txt`):
  - `transformers>=4.40.0`
  - `torch>=2.1.0`
  - `accelerate>=0.26.0`
  - `safetensors>=0.4.0`
  - `python-dotenv>=1.0.1`
  - `huggingface_hub>=0.23.0`

## Installation

1. Clone the repo:
```bash
git clone https://github.com/your-username/math-text-simplification.git
cd math-text-simplification
``` 
2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```
3. Install all required packages:
```bash
pip install -r requirements.txt
```
4. Create a .env file and add your Hugging Face token:
```bash
HF_TOKEN=hf_your_token_here
```
5. Run the Simplifier
```bash
python3 src/app.py
```

