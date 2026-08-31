import chromadb
import Retrieval.ingest as ingest
import Retrieval.chunk as chunk
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def add_to_db(research_papers_path, collection):

    research_papers_path = Path(research_papers_path)

    rp_paths = sorted(research_papers_path.rglob("*.pdf"))
    rp_metadata_path = research_papers_path / "papers_metadata.csv"
    metadata = pd.read_csv(rp_metadata_path)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    for research_paper in tqdm(rp_paths, desc=f"Storing chunk embeddings for research papers in {collection.name}"):

        pdf_text = ingest.read_pdf(research_paper)
        
        pdf_chunks = chunk.chunk_text(pdf_text)
        chunk_ids = [f"{research_paper.stem}_chunk_{i}" for i in range(len(pdf_chunks))]


        try:

            embeddings = model.encode(pdf_chunks)
        
        except TypeError as e:
            tqdm.write(f"[WARNING] {research_paper.name} : {e}")
        
        pdf_metadata = metadata[metadata["ID"] == research_paper.stem].iloc[0, :].to_dict()
        chunks_metadata = [pdf_metadata for _ in pdf_chunks]

        collection.upsert(
            ids=chunk_ids,
            documents= pdf_chunks,
            embeddings=embeddings,
            metadatas=chunks_metadata
        )


        



client = chromadb.PersistentClient(path="chroma_db")

try:

    client.delete_collection(name="research_paper_collection")
except chromadb.errors.NotFoundError as e:
    print(f"Collection could not be deleted as it does not exist. Creating Collection...")

collection = client.get_or_create_collection(
    name="research_paper_collection",
    configuration={
        "hnsw":{
            "space": "cosine",
            "ef_construction": 200,
            "ef_search": 100,
            "max_neighbors": 32
        }
    }
)

add_to_db("Retrieval/Fetch_Papers/Research_Papers", collection)
