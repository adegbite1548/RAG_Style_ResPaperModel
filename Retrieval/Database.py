import chromadb
import Retrieval.ingest as ingest
import Retrieval.chunk as chunk
from pathlib import Path
import pandas as pd
from sentence_transformers import CrossEncoder
from tqdm import tqdm
import re
import sqlite3
import uuid
import shutil
import torch

class Database:
    

    def __init__(self, client,embedding_model =None):
        self.client = client
        self.embedding_model = embedding_model
        self.reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", activation_fn=torch.nn.Sigmoid(), device="cuda")

    def create_db_collection(self, collection_name):
        collection = self.client.create_collection(
            name=collection_name,
            configuration = {
                "hnsw":{
                    "space": "cosine",
                    "ef_construction": 200,
                    "ef_search": 100,
                    "max_neighbors": 32
                }
            }
        )

        return collection
    
    def instantiate_db_collection(self, collection_name, ef_search=100):

        collection = self.client.get_collection(
            name=collection_name,
        )

        collection.modify(
            configuration={
                "hnsw": {
                    "ef_search": ef_search
                }
            }
        )

        return collection

    def add_to_db(self, 
                  research_papers_path, 
                  collection, 
                  ):

        research_papers_path = Path(research_papers_path)

        rp_paths = sorted(research_papers_path.rglob("*.pdf"))
        rp_metadata_path = research_papers_path / "papers_metadata.csv"
        metadata = pd.read_csv(rp_metadata_path)

        

        for research_paper in tqdm(rp_paths, desc=f"Storing chunk embeddings for research papers in {collection.name}"):

            existing_paper = collection.get(
                where={"ID": research_paper.stem},
                limit = 1
            )

            if existing_paper["ids"]:
                collection.delete(
                    where={"ID":research_paper.stem}
                )

            pdf_text = ingest.read_pdf(research_paper)
            
            pdf_chunks = chunk.chunk_text(pdf_text)
            chunk_ids = [f"{research_paper.stem}_chunk_{i}" for i in range(len(pdf_chunks))]


            try:

                embeddings = self.embedding_model.encode(pdf_chunks)
            
            except TypeError as e:
                tqdm.write(f"[WARNING] {research_paper.name} : {e}")
            
            pdf_metadata = metadata[metadata["ID"] == research_paper.stem].iloc[0, :].to_dict()
            chunks_metadata = [pdf_metadata for _ in pdf_chunks]

            collection.add(
                ids=chunk_ids,
                documents= pdf_chunks,
                embeddings=embeddings,
                metadatas=chunks_metadata
            )

    

    def query_db(self, collection, query):
    
        query_embeddings = self.embedding_model.encode(query)

        query_results = collection.query(
            query_embeddings = query_embeddings,
            n_results = 20,
            include=["documents", "metadatas", "distances"]
        )


        records = []

        for id, document, metadata in zip(          
                                        query_results["ids"][0], 
                                        query_results["documents"][0], 
                                        query_results["metadatas"][0]
                                        ):

            record  ={}
            record["id"] = id
            record["document"] = document
            record["metadata"] = metadata

            records.append(record)
            
        query_document_pairs = [(query, record["document"]) for record in records]

        scores = self.reranker_model.predict(query_document_pairs, batch_size = 32)

        for score, record in zip(scores, records):
            record["score"] = score


        reranked_results = sorted(
            records,
            key= lambda x: x["score"],
            reverse=True
        )
        

        return reranked_results

    def get_top_k_papers(self, results, k=5):
        seen_papers_set = set()
        document_metadata_pairs ={}

        for record in results:
            paper_id = re.split("_", record["id"])[0]

            if paper_id not in seen_papers_set:
                seen_papers_set.add(paper_id)
                document_metadata_pairs[paper_id] = (record["document"], record["metadata"])

                if len(seen_papers_set) >= k:
                    break
                
        return document_metadata_pairs
    

    def rebuild_db(self, collection, ef_construction, max_neighbours, space):
        pass

    def clear_vector_index(self, db_path):
        conn = sqlite3.connect(db_path / "chroma.sqlite3")

        active_segments = {
            row[0]
            for row in conn.execute("""
                SELECT id
                FROM segments
                WHERE scope = 'VECTOR';
            """)
        }

        conn.close()

        # Find UUID directories 
        for folder in db_path.iterdir():

            if not folder.is_dir():
                continue

            try:
                uuid.UUID(folder.name)
            except ValueError:
                continue

            # If no longer linked to chroma delete.
            if folder.name not in active_segments:
                print(f"Deleting orphaned index: {folder.name}")
                shutil.rmtree(folder)

