from flask import Flask, render_template, request, jsonify, session

from Retrieval.Database import Database
import Augmentation.Augment as aug
import Generation.Generate as gen

import chromadb
from sentence_transformers import SentenceTransformer


app = Flask(__name__)
app.secret_key = "dev-secret-key"


# -------------------------
# RAG SETUP
# -------------------------

client = chromadb.PersistentClient(
    path="chroma_db/mp_net_db"
)

database = Database(
    client,
    embedding_model=SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2"
    )
)

collection = database.instantiate_db_collection(
    "research_paper_collection_mpnet_base"
)


# -------------------------
# PAGES
# -------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chatbot")
def chatbot():
    session.pop("conversation_history", None)
    return render_template("chatbot.html")


# -------------------------
# CHAT
# -------------------------

@app.route("/ask", methods=["POST"])
def ask():

    query = request.json["query"]
    first_message = request.json["first_message"]

    conversation_history = session.get(
        "conversation_history",
        ""
    )

    # Rewrite follow-up queries
    if first_message:
        search_query = query
    else:
        search_query = gen.rewrite_query(
            query,
            conversation_history
        )

    # Retrieve papers
    retrieved_chunks = database.query_db(
        collection,
        search_query
    )

    

    top_k_unique_chunks = database.get_top_k_papers(
        retrieved_chunks
    )

    llm_context = aug.augment_query_results(
        top_k_unique_chunks
    )

    # Generate answer
    llm_prompt = gen.generate_llm_prompt(
        llm_context,
        query,
        conversation_history
    )

    llm_answer = gen.ask_llm(llm_prompt)

    # Store conversation
    conversation_history += f"""

ROLE: User
CONTENT: {query}

ROLE: Assistant
CONTENT: {llm_answer}
"""

    session["conversation_history"] = conversation_history


    return jsonify({
        "answer": llm_answer
    })


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)