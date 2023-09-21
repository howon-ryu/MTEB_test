from mteb import MTEB
from sentence_transformers import SentenceTransformer

# Define the sentence-transformers model name
#model_name = "average_word_embeddings_komninos"
# model_name = "bert-base-uncased"
# model_name = "BM-K/KoMiniLM"
# model_name = "BM-K/KoChatBART"
# model_name = "thenlper/gte-base"
#model_name = "intfloat/e5-small-v2"
# model_name = "BAAI/bge-base-en"
# model_name = "google/flan-t5-base"
# model_name = "t5-base"
# model_name = "quantumaikr/KoreanLM" # 메모리 부족 으로 인해 실패후 삭제
#model_name = "lmsys/vicuna-13b-v1.5-16k"#메모리 부족 으로 인해 실패후 삭제

#model_name = "lmsys/vicuna-7b-v1.3" #분할되어 있는 모델을 불러오지 못함... from_Tf 는 현재 SentenceTransformer 에서 삭제된기능 인데 이 오류를 리턴, checkpoint 파일의 손상이라고 얘기하나. 재설치를 해봐도 여전함


model = SentenceTransformer(model_name)
evaluation = MTEB(tasks=["Banking77Classification"])#Classification
#evaluation = MTEB(tasks=["STS16"])#STS
#evaluation = MTEB(tasks=["StackOverflowDupQuestions"])#Reranking 10분?
#evaluation = MTEB(tasks=["SummEval"])#Summarization


#evaluation = MTEB(tasks=["Tatoeba"]) # Bitext
# evaluation = MTEB(tasks=["MedrxivClusteringP2P"])#Clustering
# evaluation = MTEB(tasks=["PPC"])#Pair Classification

#evaluation = MTEB(tasks=["SweFAQ"])#Retrival





results = evaluation.run(model, output_folder=f"results/{model_name}")


print(results)