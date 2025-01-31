from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from chatbot.openai_handler import OpenAIHandler
from chatbot.vector_store import VectorStore
from chatbot.data_preparation import DataPreparation

app = FastAPI()
ai_handler = OpenAIHandler()
vector_store = VectorStore()
data_prep = DataPreparation()

class ChatRequest(BaseModel):
    question: str
    user_id: Optional[str] = "guest"  # 선택적 필드로 변경

class QAPair(BaseModel):
    id: Optional[int] = None
    index: Optional[int] = None  # 순서 번호 추가
    question: str
    answer: str

# QA 데이터 조회
@app.get("/qa/list", response_model=List[QAPair])
async def list_qa_pairs():
    try:
        collection = vector_store.create_collection()
        collection.load()
        
        results = collection.query(
            expr="id >= 0",
            output_fields=["id", "question", "answer"],
            limit=100,
            consistency_level="Strong",
            order_by="id"
        )
        
        # 1부터 시작하는 인덱스 번호 추가
        qa_pairs = [
            QAPair(
                id=item['id'],
                index=idx + 1,  # 1부터 시작하는 순서 번호
                question=item['question'],
                answer=item['answer']
            ) for idx, item in enumerate(results)
        ]
        
        collection.release()
        return qa_pairs
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# QA 데이터 추가
@app.post("/qa/add")
async def add_qa_pair(qa: QAPair):
    try:
        result = await data_prep.prepare_qa_data([{"question": qa.question, "answer": qa.answer}])
        if result:
            return {"message": "QA 쌍이 성공적으로 추가되었습니다."}
        return {"message": "QA 쌍 추가 실패"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    

# QA 데이터 삭제
@app.delete("/qa/{index}")
async def delete_qa_pair(index: int):
    try:
        collection = vector_store.create_collection()
        collection.load()
        
        # 인덱스로 실제 ID 찾기
        results = collection.query(
            expr="id >= 0",
            output_fields=["id"],
            limit=100,
            order_by="id"
        )
        
        if index < 1 or index > len(results):
            raise HTTPException(status_code=404, detail="항목을 찾을 수 없습니다.")
        
        actual_id = results[index - 1]['id']
        collection.delete(f"id == {actual_id}")
        
        collection.release()
        return {"message": f"{index}번 항목이 삭제되었습니다."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 채팅 엔드포인트
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # user_id가 없거나 "guest"인 경우에도 정상 동작
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
            context = f"질문: {results[0]['question']} f{results[0]['answer']}"
        
        # 3. OpenAI로 최종 답변 생성
        response = await ai_handler.generate_response(
            context=context,
            question=request.question
        )
        
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#uvicorn chatbot.main:app --reload