import os
from dotenv import load_dotenv
import torch
from transformers import pipeline # type: ignore

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

model_id = "meta-llama/Llama-3.1-8B-Instruct"
pipeline_llama = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    token=hf_token  #Use token from .env
)