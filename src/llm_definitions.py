from models import pipeline_llama
import json

system_message = (
    'Your task is to generate a list of up to 5 of the most difficult definitions, concepts, or equations in the following passage. '
    'The output should be in the format of a json dictionary such as: {"term1": "easy", "term2": "hard", ...}'
    "The passage will be provided in the user prompt. "
    "If there are no difficult terms, return an empty JSON object: {}."
    "Do not add extra explanation like 'Here is:' or 'The identified terms are:'."
)
def get_definitions(text):
    prompt = 'PASSAGE: ' + text + '. Identify the most difficult terms in the passage and return a JSON object with the terms and their difficulties. The difficulties can be "easy", "medium", or "hard". If no difficult terms are found, return an empty JSON object: {}'
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
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

    text = outputs[0]["generated_text"][len(prompt):].strip()
    print(text)
    return json.loads(text) if text and text != "No difficult terms found" else {}