import Retrieval.chromadb_retrieve as db_retriever
import chromadb
import re
import pandas as pd
import random
import math

############################################################
##                 Sanity Check Section                   ##
############################################################

def precision_k(collection, title_query, paper_id, k=5):

    _, query_results = db_retriever.query_db(collection, title_query)
    relevant_k = 0

    for i in range(k):
        current_id = re.split("_", query_results["ids"][0][i])[0]

        if current_id == paper_id:
            relevant_k+=1

    return relevant_k/k

def recall_k(collection, title_query, paper_id, k=5):
    total_relevant_chunks = len(collection.get(
                                where={"ID": paper_id}
                            )["ids"])

    _, query_results = db_retriever.query_db(collection, title_query)
    relevant_k = 0

    for i in range(k):
        current_id = re.split("_", query_results["ids"][0][i])[0]

        if current_id == paper_id:

            # document = query_results["documents"][0][i]
            # print(f"\n\n{document}\n\n")

            relevant_k+=1

    return relevant_k/total_relevant_chunks

def ndcg_k(collection, title_query, paper_id, k=5):
    total_relevant_chunks = len(collection.get(
                                    where={"ID": paper_id}
                                )["ids"])


    _, query_results = db_retriever.query_db(collection, title_query)
    dcg_k = 0
    ideal_dcg_k = 0

    for i in range(k):
        current_id = re.split("_", query_results["ids"][0][i])[0]

        dcg_k += 1/math.log(i+2,2) if current_id == paper_id else 0
        
        if i + 1 <= total_relevant_chunks:
            ideal_dcg_k += 1/math.log(i+2,2)
    
    ndcg_k = dcg_k/ideal_dcg_k

    return ndcg_k

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection(name="research_paper_collection")



metadata_df = pd.read_csv("Retrieval/Fetch_Papers/Research_Papers/papers_metadata.csv")




#--Calculate Precision, Recall and NDCG for single title query--

random.seed(10)
randpaper_row_index = random.randint(0, len(metadata_df)-1)
metadata_df_testsample = metadata_df.iloc[randpaper_row_index, :]
testsample_pair = (metadata_df_testsample["Title"], metadata_df_testsample["ID"])
# print(metadata_df_testsample["Title"])
k = 5
print(f"------- Precision, Recall, NDCG for single paper -------")
print(f"Precision@{k} score: {precision_k(collection, testsample_pair[0], testsample_pair[1], k = k)}")
print(f"Recall@{k} score: {recall_k(collection, testsample_pair[0], testsample_pair[1], k = k)}")
print(f"NDCG@{k} score: {ndcg_k(collection, testsample_pair[0], testsample_pair[1], k = k)}\n\n")




    