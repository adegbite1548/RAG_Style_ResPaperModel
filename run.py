from Retrieval.Database import Database
import Augmentation.Augment as aug
import chromadb
import Generation.Generate as gen
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="chroma_db/mp_net_db")

database = Database(client, embedding_model=SentenceTransformer("sentence-transformers/all-mpnet-base-v2"))

collection = database.instantiate_db_collection("research_paper_collection_mpnet_base")

query = "Recommend me research papers that have to do with robotics and planning"

retrieved_chunks = database.query_db(collection, query)

top_k_unique_chunks = database.get_top_k_papers(retrieved_chunks)

llm_context = aug.augment_query_results(top_k_unique_chunks)

print(f"\n{llm_context}\n")

llm_prompt = gen.generate_llm_prompt(llm_context, query)

print(gen.ask_llm(llm_prompt))