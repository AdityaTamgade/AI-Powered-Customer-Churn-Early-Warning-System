import pickle
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve(query, k=3):
    with open("src/rag/index.pkl", "rb") as f:
        index, docs = pickle.load(f)

    query_embedding = model.encode([query])

    D, I = index.search(query_embedding, k)

    results = [docs[i] for i in I[0]]
    return results