import ollama


def generate_llm_prompt(context, query):

    return f"""
You are a research paper recommendation assistant.

Answer the user's query using ONLY the provided context.

The context contains information about multiple research papers.
Consider ALL papers in the context before answering.
Do NOT ignore papers simply because they appear earlier or later.
Your output MUST use all papers in the context

For each recommended paper, explain why it is relevant to the query.
Do not invent paper titles, authors, or details that are not present in the context.

Context:
{context}

Query:
{query}

Answer:
"""


def ask_llm(llm_prompt, llm="llama3.1"):
    return ollama.generate(
        model=llm, 
        prompt=llm_prompt,
        options = {
            "num_ctx" : 10000
        }

        )['response']