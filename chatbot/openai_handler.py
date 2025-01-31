from openai import OpenAI
from typing import Dict, List
from dotenv import load_dotenv
import os

load_dotenv()

class OpenAIHandler:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


    # 텍스트 임베딩 생성
    async def get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            input=text, # 임베딩할 텍스트
            model="text-embedding-ada-002" # 임베딩 모델
        )
        return response.data[0].embedding # 임베딩 벡터 반환
    

    # GPT를 사용한 응답 생성
    async def generate_response(self, context: str, question: str) -> str:
        messages = [
            {"role": "system", "content": "당신은 웹 애플리케이션 지원 도우미입니다."},
            {"role": "user", "content": f"컨텍스트: {context}\n\n질문: {question}"}
        ]
        # role: system의 역할은 챗봇의 역할을 정의
        # role: user의 역할은 사용자의 질문을 정의
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
            max_tokens=500
        )
        
        return response.choices[0].message.content 