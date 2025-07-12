from fastapi import APIRouter
from app.models.news import NewsTransferRequest, NewsTransferResponse, DifficultWord, NewsTransferResult
from app.services.summarizer import build_transform_prompt, build_summary_prompt, generate_content
from app.services.difficulty_filter import build_difficult_word_prompt
from typing import List

router = APIRouter()

@router.post("/news/transfer", response_model=NewsTransferResponse)
async def transform_news(request: NewsTransferRequest):
    # 내부적으로 뉴스 유형 분류 및 모델 선택
    from app.services.mcp import classify_news_type, select_model_by_news_type
    news_type = classify_news_type(request.title, request.originalContent)
    model = select_model_by_news_type(news_type)

    # 모델에 따라 분기
    if model == "gemini":
        # Gemini API 호출 함수로 프롬프트 전달
        transformed_content = call_gemini_api(build_transform_prompt(request.title, request.originalContent, request.level))
        summary = call_gemini_api(build_summary_prompt(request.title, request.originalContent))
        difficult_words_raw = call_gemini_api(build_difficult_word_prompt(request.originalContent, request.level))
    else:
        # 로컬 모델 사용
        transformed_content = generate_content(build_transform_prompt(request.title, request.originalContent, request.level))
        summary = generate_content(build_summary_prompt(request.title, request.originalContent))
        difficult_words_raw = generate_content(build_difficult_word_prompt(request.originalContent, request.level))

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

@router.post("/news/auto_generate")
async def auto_generate_news(title: str = Body(...), content: str = Body(...), level: str = Body(...)):
    mcp_request = build_mcp_request_auto(title, content, level)
    result = await call_local_mcp(mcp_request)
    return result