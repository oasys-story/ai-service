from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import numpy as np

# Milvus 벡터 저장소 클래스
class VectorStore:
    def __init__(self):
        # Milvus 서버에 연결
        self.connect_to_milvus()
        
    def connect_to_milvus(self):
        connections.connect(
            alias="default",
            host="localhost",
            port="19530"
        )
    
    def create_collection(self, collection_name="qa_collection"):
        # 컬렉션이 없을 때만 생성
        # 있으면 기존 컬렉션 반환
        if utility.has_collection(collection_name):
            return Collection(name=collection_name)
            
        # 필드 정의
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536)
        ]
        
        # 스키마 생성
        schema = CollectionSchema(fields=fields)
        
        # 컬렉션 생성
        collection = Collection(name=collection_name, schema=schema)
        
        # 인덱스 생성
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        return collection
        
    def search_similar(self, collection_name, query_embedding, limit=3):
        # 매 검색 요청마다 실행
        # 벡터 유사도 검색 수행
        collection = Collection(name=collection_name)
        collection.load()
        
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            output_fields=["question", "answer"]
        )
        
        # 검색 결과 형식 변환
        hits = []
        for hit in results[0]:  # 첫 번째 쿼리의 결과
            hits.append({
                "question": hit.entity.get('question'),
                "answer": hit.entity.get('answer')
            })
        
        return hits 