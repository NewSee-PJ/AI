# app/services/mcp.py

from app.models.mcp import MCPRequest, MCPRequestItem
from app.services.summarizer import build_transform_prompt

def classify_news_type(title: str, content: str) -> str:
    keywords_politics = ["총선", "외교", "갈등", "정책", "정치", "국회", "대통령", "정부", "외교부"]
    text = (title + " " + content)
    if any(keyword in text for keyword in keywords_politics):
        return "정치, 외교, 사회 이슈"
    return "기타"

def select_model_by_news_type(news_type: str) -> str:
    if news_type == "정치, 외교, 사회 이슈":
        return "gemini"
    else:
        return "kullm3"

def build_mcp_request_auto(title: str, content: str, level: str) -> MCPRequest:
    news_type = classify_news_type(title, content)
    model = select_model_by_news_type(news_type)
    prompt = build_transform_prompt(title, content, level)
    item = MCPRequestItem(prompt=prompt, model=model, metadata={"news_type": news_type, "level": level})
    return MCPRequest(items=[item])
