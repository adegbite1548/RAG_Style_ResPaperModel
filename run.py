import Retrieval.chromadb_retrieve as db_retreiver
import Augmentation.Augment as aug
import chromadb
import Generation.Generate as gen

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection(name="research_paper_collection")

query = "Recommend me research papers that have to do with robotics and planning"
llm_context = aug.augment_query_results(db_retreiver.query_db(collection, query)[0])

print(f"\n{llm_context}\n")

llm_prompt = gen.generate_llm_prompt(llm_context, query)

print(gen.ask_llm(llm_prompt))