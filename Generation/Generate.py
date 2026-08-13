import ollama


def generate_llm_prompt(context, query):

    return f"Answer the query based only on the following context: {context}\nQuery: {query}"


def ask_llm(llm_prompt, llm="llama3.1"):
    return ollama.generate(model=llm, prompt=llm_prompt)['response']