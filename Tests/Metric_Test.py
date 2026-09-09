from Retrieval.Database import Database
import chromadb
import re
import pandas as pd
import random
import math
from sentence_transformers import SentenceTransformer



def chunk_precision_k(database, collection_name, title_query, paper_id, k=5):

    collection = database.instantiate_db_collection(collection_name)

    results = database.query_db(collection, title_query)
    relevant_k = 0

    for i in range(k):
        current_id = re.split("_", results[i]["id"])[0]

        if current_id == paper_id:
            relevant_k+=1

    return relevant_k/k

def chunk_recall_k(database, collection_name,title_query, paper_id, k=5):
    collection = database.instantiate_db_collection(collection_name)

    total_relevant_chunks = len(collection.get(
                                where={"ID": paper_id}
                            )["ids"])

    results = database.query_db(collection, title_query)
    relevant_k = 0

    for i in range(k):
        current_id = re.split("_", results[i]["id"])[0]

        if current_id == paper_id:

            # document = query_results["documents"][0][i]
            # print(f"\n\n{document}\n\n")

            relevant_k+=1

    return relevant_k/total_relevant_chunks

def chunk_ndcg_k(database, collection_name, title_query, paper_id, k=5):

    collection = database.instantiate_db_collection(collection_name)

    total_relevant_chunks = len(collection.get(
                                    where={"ID": paper_id}
                                )["ids"])


    results = database.query_db(collection, title_query)
    dcg_k = 0
    ideal_dcg_k = 0


    for i in range(k):
        current_id = re.split("_", results[i]["id"])[0]

        dcg_k += 1/math.log(i+2,2) if current_id == paper_id else 0
        
        if i + 1 <= total_relevant_chunks:
            ideal_dcg_k += 1/math.log(i+2,2)
    
    ndcg_k = dcg_k/ideal_dcg_k

    return ndcg_k

client = chromadb.PersistentClient(path="chroma_db/mp_net_db")
database = Database(client, embedding_model=SentenceTransformer("sentence-transformers/all-mpnet-base-v2"))
collection_name = "research_paper_collection_mpnet_base"

metadata_df = pd.read_csv("Retrieval/Fetch_Papers/Research_Papers/papers_metadata.csv")




#--Calculate Precision, Recall and NDCG for single title query--

random.seed(10)

num_papers = 50


randpaper_row_indexes = random.sample(range(len(metadata_df)), num_papers)

metadata_df_testsamples = metadata_df.iloc[randpaper_row_indexes, :]
testsample_pairs = (metadata_df_testsamples["Title"], metadata_df_testsamples["ID"])


k_values = [1,2,5,10]
metric_sums = []

for k in k_values:
    sum_precision_k = 0
    sum_recall_k = 0
    sum_ndcg_k = 0
    for test_title, test_ID in zip(metadata_df_testsamples["Title"], metadata_df_testsamples["ID"]): 
        sum_precision_k += chunk_precision_k(database, collection_name, test_title, test_ID, k = k)
        sum_recall_k += chunk_recall_k(database, collection_name, test_title, test_ID, k = k)
        sum_ndcg_k += chunk_ndcg_k(database, collection_name, test_title, test_ID, k = k)

    metric_sums.append((sum_precision_k, sum_recall_k, sum_ndcg_k))


for k, metric_sum in zip(k_values, metric_sums):
    print(f"\n")
    print(f"------Printing mean chunk Recall, Precision and NDCG for {num_papers} papers @k={k}---------")
    print(f"\n")
    print(f"Mean chunk Precision@{k} : {metric_sum[0]/num_papers}")
    print(f"Mean chunk Recall@{k} : {metric_sum[1]/num_papers}")
    print(f"Mean chunk NDCG@{k} : {metric_sum[2]/num_papers}")
    print(f"\n")
    print(f"-------------------------------------------------------------------------------------------")
    print(f"\n")





    