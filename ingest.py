from sentence_transformers import SentenceTransformer
import chromadb
from pathlib import Path

#embedding model
model = SentenceTransformer("all-MiniLM-L6-V2")

#chroma client
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="cti_docs")

data_folder = Path("data")

for file in data_folder.glob("*.txt"):
    text = file.read_text(encoding="utf-8")
    embedding = model.encode(text).tolist()

    collection.add(
        documents=[text],
        embeddings=[embedding],
        ids=[file.stem]

    )

print("Documents loaded into Chroma.")
                                             