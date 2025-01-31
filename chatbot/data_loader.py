import asyncio
from chatbot.data_preparation import DataPreparation, prepare_qa_dataset

async def main():
    # data_preparation.py의 prepare_qa_dataset() 함수에서 QA 데이터 가져옴
    qa_pairs = prepare_qa_dataset()
    
    # DataPreparation 클래스를 사용해서 데이터를 Milvus에 저장
    data_prep = DataPreparation()
    result = await data_prep.prepare_qa_data(qa_pairs)
    
    if result:
        print(f"총 {len(qa_pairs)}개의 QA 데이터가 성공적으로 저장되었습니다.")
    else:
        print("데이터 저장 중 오류가 발생했습니다.")

if __name__ == "__main__":
    asyncio.run(main()) 