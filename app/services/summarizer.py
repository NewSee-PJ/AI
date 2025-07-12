# 비즈니스 로직 / AI 추론 모듈

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 모델과 토크나이저를 전역으로 로드 (최초 1회만)
tokenizer = AutoTokenizer.from_pretrained("nlpai-lab/KULLM3")
model = AutoModelForCausalLM.from_pretrained("nlpai-lab/KULLM3")

def generate_content(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 프롬프트 부분 제거 (모델에 따라 필요)
    return result[len(prompt):].strip() if result.startswith(prompt) else result

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
