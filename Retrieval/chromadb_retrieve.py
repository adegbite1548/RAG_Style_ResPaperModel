from sentence_transformers import SentenceTransformer
import re

def query_db(collection, query, model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")):

    query_embeddings = model.encode(query)

    query_results = collection.query(
        query_embeddings = query_embeddings,
        n_results = 100,
        include=["documents", "metadatas", "distances"]
    )

    seen_papers_set = set()
    combined_results = {}

    for id, document, distance, metadata in zip(query_results["ids"][0], 
                                                query_results["documents"][0], 
                                                query_results["distances"][0], 
                                                query_results["metadatas"][0]
                                                ):

        paper_id = re.split("_", id)[0]

        


        if paper_id not in seen_papers_set:
            seen_papers_set.add(paper_id)
            combined_results[paper_id] = (document, metadata, distance)

            if len(seen_papers_set) >= 5:
                break

            
    
    return combined_results, query_results
    
   
















