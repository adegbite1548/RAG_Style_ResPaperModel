from Retrieval.Database import Database 
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
import os

db_path = "chroma_db/mp_net_db"

os.makedirs(db_path, exist_ok=True)

client_mpnet = chromadb.PersistentClient(path=db_path)

database_mpnet = Database(client_mpnet, embedding_model=SentenceTransformer("sentence-transformers/all-mpnet-base-v2"))

collection = database_mpnet.instantiate_db_collection("research_paper_collection_mpnet_base")






###################################
## Section to add to db ###########
###################################

database_mpnet.add_to_db("Retrieval/Fetch_Papers/Research_Papers", collection)

