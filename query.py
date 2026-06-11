from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-V2")

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection("cti_docs")

question = input("Question: ")

query_embedding = model.encode(question).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=1 # amount of chunks to return
)

print("\nRetrieved Chunks:\n")

for doc in results["documents"][0]:
    print(doc)
    print("-" * 50)