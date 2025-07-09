import os
from dotenv import load_dotenv
import torch
from transformers import pipeline

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
pipeline_llama = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    token=hf_token  #Use token from .env
)

system_message = (
    "Your task is to simplify scientific sentences into an easy-to-read sentence "
    "while keeping the main content. Do not add extra explanation like 'Here is:'"
)

def simplify_math_text(text):
    user_message = f"Simplify this sentence: SENTENCE: {text}. Simplify the SENTENCE."
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    prompt = pipeline_llama.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    terminators = [
        pipeline_llama.tokenizer.eos_token_id,
        pipeline_llama.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline_llama(
        prompt,
        max_new_tokens=200,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        pad_token_id=pipeline_llama.tokenizer.eos_token_id,
    )

    return outputs[0]["generated_text"][len(prompt):].strip()
