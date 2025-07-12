# 비즈니스 로직 / AI 추론 모듈

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
