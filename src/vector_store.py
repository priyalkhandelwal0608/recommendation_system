import faiss
import numpy as np

def build_index():
    embeddings = np.load("models/item_embeddings.npy")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, "models/faiss_index.index")
    print("Vector index created")

def recommend(user_embedding, top_k=3):
    index = faiss.read_index("models/faiss_index.index")
    distances, indices = index.search(user_embedding, top_k)
    return indices