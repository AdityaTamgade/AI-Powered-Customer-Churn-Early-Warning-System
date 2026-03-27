from sentence_transformers import SentenceTransformer
import faiss
import pickle

model = SentenceTransformer('all-MiniLM-L6-v2')

def create_vector_store():
    with open("src/rag/knowledge.txt", "r") as f:
        docs = f.readlines()

    docs = [doc.strip() for doc in docs if doc.strip()]

    embeddings = model.encode(docs)

    index = faiss.IndexFlatL2(384)
    index.add(embeddings)

    with open("src/rag/index.pkl", "wb") as f:
        pickle.dump((index, docs), f)

    print("✅ Vector DB created!")


if __name__ == "__main__":
    create_vector_store()