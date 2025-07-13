# app/services/rag.py

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Quadrant
from sentence_transformers import SentenceTransformer

# 1. 임베딩 모델 준비
embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-small",
    model_kwargs={"device": "cpu"}  # GPU 사용시 "cuda"
)

# 2. 쿼드런트DB 벡터스토어 연결 (환경변수로 API키 등 설정 필요)
vectorstore = Quadrant(
    embedding=embedding_model,
    collection_name="dictionary",  # 사전 데이터가 저장된 컬렉션명
    url="https://api.cloud.quadrant.io",  # 쿼드런트DB 엔드포인트
    api_key="YOUR_QUADRANT_API_KEY"
)

def search_word_info(word: str, top_k: int = 1) -> str:
    # 쿼드런트DB에서 임베딩 기반으로 단어 설명 검색
    docs = vectorstore.similarity_search(word, k=top_k)
    if docs:
        return docs[0].page_content  # 가장 유사한 설명 반환
    return "설명을 찾을 수 없습니다."

async def process_news_with_rag(title, content, level):
    # 1. 어려운 단어 추출 (기존 LLM 활용)
    word_prompt = build_difficult_word_prompt(content, level)
    difficult_words_raw = await generate_content(word_prompt)
    difficult_words = []
    for line in difficult_words_raw.splitlines():
        if ":" in line:
            term, _ = line.split(":", 1)
            difficult_words.append(term.strip())

    # 2. RAG로 단어 설명 검색 (레벨 하일 때만)
    wordbook = []
    if level == "하":
        for word in difficult_words:
            info = search_word_info(word)
            wordbook.append({"term": word, "description": info})
    else:
        # 기존 방식(LLM 설명) 사용
        for line in difficult_words_raw.splitlines():
            if ":" in line:
                term, description = line.split(":", 1)
                wordbook.append({"term": term.strip(), "description": description.strip()})

    return wordbook