from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chatbot.openai_handler import OpenAIHandler
from chatbot.vector_store import VectorStore

app = FastAPI()
ai_handler = OpenAIHandler()
vector_store = VectorStore()

class ChatRequest(BaseModel):
    question: str
    user_id: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # 1. 질문에 대한 임베딩 생성
        question_embedding = await ai_handler.get_embedding(request.question)
        
        # 2. Milvus에서 유사한 QA 검색
        collection = vector_store.create_collection()
        results = vector_store.search_similar(
            collection_name="qa_collection",
            query_embedding=question_embedding,
            limit=1
        )
        
        if not results:
            context = "이 서비스는 게시판 기능을 제공하는 웹 애플리케이션입니다."
        else:
            context = f"질문: {results[0]['question']}\n답변: {results[0]['answer']}"
        
        # 3. OpenAI로 최종 답변 생성
        response = await ai_handler.generate_response(
            context=context,
            question=request.question
        )
        
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/test")
async def test_chat(request: ChatRequest):
    try:
        response = await ai_handler.generate_response(
            context="이 서비스는 게시판 기능을 제공하는 웹 애플리케이션입니다.",
            question=request.question
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 