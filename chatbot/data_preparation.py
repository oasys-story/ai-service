from typing import List, Dict
import json
from chatbot.openai_handler import OpenAIHandler
from chatbot.vector_store import VectorStore


class DataPreparation:
    def __init__(self):
        self.ai_handler = OpenAIHandler()
        self.vector_store = VectorStore()
        
    # 데이터 준비
    async def prepare_qa_data(self, qa_list: List[Dict[str, str]]):
        """
        QA 데이터를 준비하고 Milvus에 저장
        qa_list 형식: [{"question": "질문내용", "answer": "답변내용"}, ...]
        """
        collection = self.vector_store.create_collection()
        collection.load()
        
        # 데이터 삽입을 위한 리스트 준비
        data = []
        for qa in qa_list:
            embedding = await self.ai_handler.get_embedding(qa["question"])
            data.append({
                "question": qa["question"],
                "answer": qa["answer"],
                "embedding": embedding
            })
        
        try:
            # 한 번에 모든 데이터 삽입
            mr = collection.insert(data)
            collection.flush()
            
            # ID 확인을 위한 쿼리
            results = collection.query(
                expr=f"id in {mr.primary_keys}",
                output_fields=["id", "question", "answer"],
                consistency_level="Strong"
            )
            
            print("Inserted data with IDs:")
            for result in results:
                print(f"ID: {result['id']}, Question: {result['question']}")
            
        finally:
            collection.release()
        
        return True

# QA 데이터 준비 함수
def prepare_qa_dataset():
    qa_pairs = [
        {
            "question": "게시판은 어떻게 이용하나요?",
            "answer": "상단 메뉴에서 '게시판'을 클릭하시면 문의사항과 공지사항을 확인하실 수 있습니다."
        },
        {
            "question": "새 글을 작성하려면 어떻게 해야 하나요?",
            "answer": "'글쓰기' 버튼을 클릭하여 제목과 내용을 입력한 후 '등록' 버튼을 누르시면 됩니다."
        },
        {
            "question": "문의사항에 대한 답변은 어떻게 확인하나요?",
            "answer": "작성하신 문의글에 답변이 달리면 이메일로 알림이 발송되며, 게시판에서 직접 확인도 가능합니다."
        },
        {
            "question": "첨부파일은 어떻게 업로드 하나요?",
            "answer": "글 작성 시 하단의 '파일 첨부' 버튼을 클릭하여 파일을 선택하시면 됩니다. 최대 10MB까지 업로드 가능합니다."
        },
        {
            "question": "게시글을 수정하고 싶어요",
            "answer": "본인이 작성한 게시글의 경우, 게시글 하단의 '수정' 버튼을 클릭하여 내용을 수정할 수 있습니다."
        }
    ]

    return qa_pairs 