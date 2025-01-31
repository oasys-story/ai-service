from vector_store import VectorStore
from pymilvus import Collection

# Milvus에 저장된 데이터 확인
def check_milvus_data():
    # Milvus 연결
    vector_store = VectorStore()
    
    # 컬렉션 가져오기
    collection = Collection("qa_collection")
    collection.load()
    
    # 전체 데이터 조회
    results = collection.query(
        expr="id >= 0",  # 모든 데이터 조회
        output_fields=["question", "answer"],
        limit=100
    )
    
    print(f"총 {len(results)}개의 데이터가 저장되어 있습니다.")
    for idx, item in enumerate(results, 1):
        print(f"\n{idx}번째 데이터:")
        print(f"질문: {item['question']}")
        print(f"답변: {item['answer']}")

if __name__ == "__main__":
    check_milvus_data() 