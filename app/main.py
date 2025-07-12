from fastapi import FastAPI
from app.api.v1.news_transform import router

app = FastAPI()
app.include_router(router, prefix="/api/news")