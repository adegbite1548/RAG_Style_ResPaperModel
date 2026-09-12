from Retrieval.Database import Database
import Augmentation.Augment as aug
import chromadb
import Generation.Generate as gen
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="chroma_db/mp_net_db")

database = Database(client, embedding_model=SentenceTransformer("sentence-transformers/all-mpnet-base-v2"))

collection = database.instantiate_db_collection("research_paper_collection_mpnet_base")

print("######## Research Paper Recommender Prototype ########\n")
query = input("Enter the type of research paper you would like to get recommendations for: ")


conversation_history = []

if query.strip().lower() != "exit":


    while True:

        retrieved_chunks = database.query_db(collection, query)
        top_k_unique_chunks = database.get_top_k_papers(retrieved_chunks)

        llm_context = aug.augment_query_results(top_k_unique_chunks)
        print(llm_context)
        llm_prompt = gen.generate_llm_prompt(llm_context, query)
        llm_answer = gen.ask_llm(llm_prompt)

        conversation_history.append({
            "role": "user",
            "content": query
        })

        conversation_history.append({
            "role": "llm",
            "content": llm_answer
        })

        print(llm_answer)

        follow_up = input("\nEnter a follow up prompt if you would like to fine tune your search: ")

        if follow_up.strip().lower() == "exit":
            break

        query = gen.rewrite_query(follow_up, conversation_history)
        print(f"Query: {query}")

   
# Recommend me research papers linked to neural ODE's in robotics

    
    