from sentence_transformers import SentenceTransformer

def query_db(collection, query, model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")):

    query_embeddings = model.encode(query)

    query_results = collection.query(
        query_embeddings = query_embeddings,
        n_results = 5
        
    )

    combined_results = []

    for document, metadata in zip(query_results["documents"][0], query_results["metadatas"][0]):
        combined_results.append((document, metadata))

   
    return combined_results
    
   
















