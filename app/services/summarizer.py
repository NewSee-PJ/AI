# app/services/summarizer.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
from typing import List
import httpx

# 모델 로딩 및 4bit 양자화 설정
model_id = "sunnyanna/KULLM3-AWQ"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

def build_chat_prompt(prompt: str):
    return f"<s>[INST] {prompt.strip()} [/INST]"

def kullm_batch_generate(prompts: List[str], max_new_tokens=512):
    chat_prompts = [build_chat_prompt(p) for p in prompts]
    inputs = tokenizer(chat_prompts, return_tensors="pt", padding=True).to(model.device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.2,
        top_p=0.2,
        pad_token_id=tokenizer.eos_token_id
    )
    decoded_results = []
    for i in range(len(prompts)):
        result_text = tokenizer.decode(output[i], skip_special_tokens=True)
        decoded_results.append(result_text.split('[/INST]')[-1].strip())
    return decoded_results

VLLM_API_URL = "http://localhost:8000/v1/completions"

async def vllm_generate_content(prompt: str, max_tokens: int = 512) -> str:
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "sunnyanna/KULLM3-AWQ",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "top_p": 0.2,
        "stop": None
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(VLLM_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["text"].strip()

# 기존 generate_content 함수 대체
async def generate_content(prompt: str, max_new_tokens=512) -> str:
    return await vllm_generate_content(prompt, max_tokens=max_new_tokens)