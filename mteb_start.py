from mteb import MTEB
from sentence_transformers import SentenceTransformer

# Define the sentence-transformers model name
#model_name = "average_word_embeddings_komninos"
# model_name = "bert-base-uncased"
# model_name = "BM-K/KoMiniLM"
# model_name = "BM-K/KoChatBART"
# model_name = "thenlper/gte-base"
#model_name = "intfloat/e5-small-v2"
model_name = "BAAI/bge-base-en"
model = SentenceTransformer(model_name)
evaluation = MTEB(tasks=["Banking77Classification"])
results = evaluation.run(model, output_folder=f"results/{model_name}")


print(results)