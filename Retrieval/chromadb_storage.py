import chromadb
import ingest
import chunk
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer

def add_to_db(research_papers_path, collection):

    research_papers_path = Path(research_papers_path)

    rp_paths = sorted(research_papers_path.rglob("*.pdf"))
    rp_metadata_path = research_papers_path / "papers_metadata.csv"
    metadata = pd.read_csv(rp_metadata_path)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    for research_paper in rp_paths:

        pdf_text = ingest.read_pdf(research_paper)
        
        pdf_chunks = chunk.chunk_text(pdf_text)
        chunk_ids = [f"{research_paper.stem}_chunk_{i}" for i in range(len(pdf_chunks))]

        embeddings = model.encode(pdf_chunks)
        
        
        pdf_metadata = metadata[metadata["ID"] == research_paper.stem].iloc[0, 1:].to_dict()
        chunks_metadata = [pdf_metadata for chunks in pdf_chunks]

        collection.upsert(
            ids=chunk_ids,
            documents= pdf_chunks,
            embeddings=embeddings,
            metadatas=chunks_metadata
        )


        



client = chromadb.PersistentClient(path="../chroma_db")
client.delete_collection(name="research_paper_collection")

collection = client.get_or_create_collection(name="research_paper_collection")

add_to_db("Fetch_Papers/Research_Papers", collection)
