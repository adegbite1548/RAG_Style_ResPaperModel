import ollama


def generate_llm_prompt(context, query):

    return f"""
You are a research paper ranker.

Answer the user's query using ONLY the provided context.

The context contains information about multiple research papers.
Consider ALL papers in the context before answering.
Do NOT ignore papers simply because they appear earlier or later.


For each ranked paper, explain why it is relevant to the query.
Do not invent paper titles, authors, or details that are not present in the context.

You should RANK the papers, not discard them simply because they are not highly relevant.
Additionally, you can have another section specifically for these other papers that are not highly relevant.

Do not require the paper's primary objective to exactly match the query.

A paper can still be relevant if it contains experiments, methods,
results, or discussion that provide useful evidence about the query.

However, do not consider a paper relevant merely because it shares
general concepts or keywords with the query.

Your Answer MUST be numbered and take the following format for each ranking:
    - [ACTUAL PAPER TITLE]
    - Reason: [Brief explanation of why this paper is relevant to the query.]


Context:
{context}

Query:
{query}
"""


def ask_llm(llm_prompt, llm="llama3:8b"):
    return ollama.generate(
        model=llm, 
        prompt=llm_prompt,
        options = {
            "num_ctx" : 10000
        }

        )['response']

def rewrite_query(follow_up, conversation_history, llm="llama3:8b"):

    prompt = f"""
You are a research query rewriting assistant.

Your task is to rewrite the user's FOLLOW-UP REQUEST into a single,
standalone search query for a research paper retrieval system.

Use the CONVERSATION HISTORY to understand what the user is referring to.

IMPORTANT RULES:

- Preserve the user's original research topic when the follow-up depends
  on previous messages.
- Resolve references such as "these", "those", "the second one",
  "more like this", "ones about X", etc. using the conversation history.
- Incorporate new requirements or changes introduced by the follow-up.
- Do not remove important concepts from the original query unless the
  user explicitly changes them.
- Do not add requirements that the user did not ask for.
- The rewritten query must be understandable WITHOUT the conversation history.
- Write ONLY the rewritten search query.
- Do NOT explain your reasoning.
- Do NOT answer the user's request.
- Do NOT recommend papers.
- Do NOT use quotation marks around the query.

CONVERSATION HISTORY:
{conversation_history}

FOLLOW-UP REQUEST:
{follow_up}

REWRITTEN SEARCH QUERY:
"""

    return ollama.generate(
        model=llm,
        prompt=prompt,
        options={
            "num_ctx": 10000
        }
    )['response'].strip()