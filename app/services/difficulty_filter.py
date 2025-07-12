def build_difficult_word_prompt(content: str, level: str) -> str:
    if level == "상":
        instruction = "다음 뉴스 본문에서 성인이 어려워할 수 있는 단어 3~5개를 골라줘. 각 단어의 뜻을 전문적으로 설명해줘."
    elif level == "중":
        instruction = "다음 뉴스 본문에서 중학생이 어려워할 수 있는 단어 5~10개를 골라줘. 각 단어의 뜻을 중학생 눈높이에 맞게 설명해줘."
    else:
        instruction = "다음 뉴스 본문에서 초등학생이 어려워할 수 있는 단어 5~10개를 골라줘. 각 단어의 뜻을 아주 쉽게 설명해줘."
    return f"{instruction}\n\n뉴스 본문:\n{content}"