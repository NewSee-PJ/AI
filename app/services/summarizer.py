# app/services/summarizer.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
from typing import List

# 모델 로딩 및 4bit 양자화 설정
model_id = "nlpai-lab/KULLM3"
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

def generate_content(prompt: str, max_new_tokens=512) -> str:
    return kullm_batch_generate([prompt], max_new_tokens=max_new_tokens)[0]