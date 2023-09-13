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
model_name = "quantumaikr/KoreanLM"

model = SentenceTransformer(model_name)
#evaluation = MTEB(tasks=["Banking77Classification"])#Classification
# evaluation = MTEB(tasks=["BUCC"]) # Bitext
#evaluation = MTEB(tasks=["MedrxivClusteringP2P"])#Clustering
# evaluation = MTEB(tasks=["TwitterURLCorpus"])#Pair Classification
# evaluation = MTEB(tasks=["MindSmallReranking"])#Reranking
# evaluation = MTEB(tasks=["ArguAna"])#Retrival
# evaluation = MTEB(tasks=["BIOSSES"])#STS
# evaluation = MTEB(tasks=["SummEval"])#Summarization
results = evaluation.run(model, output_folder=f"results/{model_name}")


print(results)