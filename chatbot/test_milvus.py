from vector_store import VectorStore

# Milvus 연결 테스트
def test_connection():
    try:
        vs = VectorStore()
        collection = vs.create_collection()
        print("Milvus 연결 성공!")
        print(f"컬렉션 생성 완료: {collection.name}")
        return True
    except Exception as e:
        print(f"연결 실패: {str(e)}")
        return False

if __name__ == "__main__":
    test_connection() 