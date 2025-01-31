from pymilvus import utility, connections
import asyncio

# Milvus 컬렉션 초기화
def reset_milvus_collection():
    # Milvus 연결
    connections.connect(
        alias="default",
        host="localhost",
        port="19530"
    )
    
    # 컬렉션 존재 여부 확인 및 삭제
    if utility.has_collection("qa_collection"):
        utility.drop_collection("qa_collection")
        print("기존 qa_collection 삭제 완료")
    else:
        print("삭제할 qa_collection이 없습니다.")
    
    # Milvus 연결 종료
    connections.disconnect("default")

if __name__ == "__main__":
    reset_milvus_collection() 