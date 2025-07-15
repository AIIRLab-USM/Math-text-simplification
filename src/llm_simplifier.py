import json
from models import pipeline_llama

system_message = (
    "Your task is to simplify scientific and mathematic passages into an easy-to-read passage "
    "You will be provided a json object with the following keys: 'term1', 'term2', etc. and their corresponding difficulties. "
    "The difficulties can be 'easy', 'medium', or 'hard'. In the passage, you must replace the complex terms with simpler alternatives or add definitions for them. "
    "You must keep the main content, and you should not try to answer any questions, or solve any equations. Do not add extra explanation like 'Here is:'"
)

def simplify_math_text(text, terms):
    terms_json = json.dumps(terms)
    user_prompt = f"PASSAGE: {text}. Simplify the passage by replacing complex terms with simpler alternatives, and add the definitions and simplifications for the identified terms. The terms you must define are: {terms_json}. The passage should still convey the main content and not try to answer any questions or solve any equations."
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]

    prompt = pipeline_llama.tokenizer.apply_chat_template( # type: ignore
        messages, tokenize=False, add_generation_prompt=True
    )

    terminators = [
        pipeline_llama.tokenizer.eos_token_id, # type: ignore
        pipeline_llama.tokenizer.convert_tokens_to_ids("<|eot_id|>") # type: ignore
    ]

    outputs = pipeline_llama(
        prompt, # type: ignore
        max_new_tokens=200,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        pad_token_id=pipeline_llama.tokenizer.eos_token_id, # type: ignore
    ) # type: ignore

    return outputs[0]["generated_text"][len(prompt):].strip()
