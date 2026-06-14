from sentence_transformers import SentenceTransformer
import chromadb
import ollama

#embedding model using library instead of building out neural netowrk
model = SentenceTransformer("all-MiniLM-L6-V2") #all general-purpose nueral network embedding model with 6 transformer layers

#declare ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

#assign collection from the Chroma DB that has our target embeddings
collection = client.get_collection("cti_docs")
#assign question
question = input("Question: ")
#embed our question to embedding mode lto get the vector
query_embedding = model.encode(question).tolist()
#send question to the chromadb to check for similarity to any of the other vectors, return 1 result
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=1 # amount of chunks to return
)
#print retrieved documents
context = results["documents"][0][0]
print("\nRetrieved Context:\n")
print(context)
print("-" * 50)
#create prompt to run the retrieved context trough the LLM to generate answer to the question from the retireved oontext
prompt = f""" 
You are a helpful assistant. Answer the question using only the context below.

Context:
{context}

Question:
{question}

Answer:
"""
#initiate Ollama chat ith gamme2:9b and send messageto generate and store response
response = ollama.chat(
    model ="gemma2:9b",
    messages=[
        {"role": "user", "content": prompt}
         
    ]
)
#print the LLM response
print("\nLLM Answer:\n")
print(response["message"]["content"])

