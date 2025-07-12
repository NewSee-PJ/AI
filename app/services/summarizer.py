# 비즈니스 로직 / AI 추론 모듈

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
from typing import List

# 1. 모델 불러오기 및 4bit 양자화 설정
model_id = "nlpai-lab/KULLM3"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"Initial VRAM usage: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")

start_load_time = time.time()
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
end_load_time = time.time()

print(f"\nModel loaded in {end_load_time - start_load_time:.2f} seconds")

if torch.cuda.is_available():
    initial_vram_after_load = torch.cuda.memory_allocated()
    peak_vram_after_load = torch.cuda.max_memory_allocated()
    print(f"VRAM allocated after model load: {initial_vram_after_load / (1024**3):.2f} GB")
    print(f"Peak VRAM used during model load: {peak_vram_after_load / (1024**3):.2f} GB")

# 2. LLM 채팅 프롬프트 포맷

def build_chat_prompt(prompt: str):
    return f"<s>[INST] {prompt.strip()} [/INST]"

# 3. 프롬프트 생성 함수 (기존 유지)
def build_transform_prompt(title: str, content: str, level: str) -> str:
    base = f"다음 뉴스 제목과 본문을 사용자의 이해 수준에 맞게 다시 써줘.\n\n뉴스 제목: {title}\n뉴스 본문: {content}\n"
    if level == "상":
        instruction = "원문에 가깝게 유지해."
    elif level == "중":
        instruction = "간결하고 이해하기 쉬운 문장으로 재구성해줘. 핵심 내용만 유지해도 좋아."
    else:
        instruction = "초등학생도 이해할 수 있도록 아주 쉽게 설명해줘. 쉬운 단어와 짧은 문장을 써줘."
    return base + "\n요청 사항: " + instruction

def build_summary_prompt(title: str, content: str) -> str:
    return f"다음 뉴스 제목과 본문을 한문장으로 간단히 요약해줘.\n\n뉴스 제목: {title}\n뉴스 본문: {content}"

# 4. 배치 추론 함수
def kullm_batch_generate(prompts: List[str], max_new_tokens=512):
    chat_prompts = [build_chat_prompt(p) for p in prompts]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    inputs = tokenizer(chat_prompts, return_tensors="pt", padding=True).to(model.device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    start_infer_time = time.time()
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.2,
        top_p=0.2,
        pad_token_id=tokenizer.eos_token_id
    )
    end_infer_time = time.time()
    generation_time = end_infer_time - start_infer_time
    decoded_results = []
    generated_tokens_list = []
    for i in range(len(prompts)):
        original_input_len = (input_ids[i] != tokenizer.pad_token_id).sum().item()
        generated_tokens = output[i].shape[0] - original_input_len
        generated_tokens_list.append(generated_tokens)
        result_text = tokenizer.decode(output[i], skip_special_tokens=True)
        decoded_results.append(result_text.split('[/INST]')[-1].strip())
    current_vram = 0
    peak_vram = 0
    if torch.cuda.is_available():
        current_vram = torch.cuda.memory_allocated()
        peak_vram = torch.cuda.max_memory_allocated()
    return decoded_results, generation_time, generated_tokens_list, current_vram, peak_vram

# 5. 단일 프롬프트용 generate_content 함수
def generate_content(prompt: str, max_new_tokens=512) -> str:
    results, _, _, _, _ = kullm_batch_generate([prompt], max_new_tokens=max_new_tokens)
    return results[0]
