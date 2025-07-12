from fastapi import APIRouter
from app.models.news import NewsTransferRequest, NewsTransferResponse, DifficultWord, NewsTransferResult
from app.services.summarizer import build_transform_prompt, build_summary_prompt, generate_content
from app.services.difficulty_filter import build_difficult_word_prompt
from typing import List

router = APIRouter()

@router.post("/transfer", response_model=NewsTransferResponse)
async def transform_news(request: NewsTransferRequest):
    transform_prompt = build_transform_prompt(request.title, request.originalContent, request.level)
    summary_prompt = build_summary_prompt(request.title, request.originalContent)
    word_prompt = build_difficult_word_prompt(request.originalContent, request.level)

    transformed_content = generate_content(transform_prompt)
    summary = generate_content(summary_prompt)
    difficult_words_raw = generate_content(word_prompt)

    difficult_words = []
    for line in difficult_words_raw.splitlines():
        if ":" in line:
            term, description = line.split(":", 1)
            difficult_words.append(DifficultWord(term=term.strip(), description=description.strip()))

    return NewsTransferResponse(
        code="NEWS200",
        message="난이도별 뉴스 변환 및 단어 추출 성공",
        result=NewsTransferResult(
            level=request.level,
            title=request.title,
            transformedContent=transformed_content,
            summarized=summary,
            difficultWords=difficult_words
        ),
        success=True
    )