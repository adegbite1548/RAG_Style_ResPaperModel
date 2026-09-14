import ollama


def generate_llm_prompt(context, query, conversation_history):

    return f"""
You are a research paper ranker.

Answer the user's query using ONLY the provided paper titles and context.

The context contains information about multiple research papers.
Consider ALL papers in the context before answering.
Do NOT ignore papers simply because they appear earlier or later.


For each ranked paper, explain why it is relevant to the query.
Do not invent paper titles, authors, or details that are not present in the context.

You should RANK the papers, not discard them simply because they are not highly relevant.
Additionally, you can have another section specifically for these other papers that are not highly relevant.


A paper can still be relevant if it contains experiments, methods,
results, or discussion that provide useful evidence about the query.

However, do not consider a paper relevant merely because it shares
general concepts or keywords with the query.

Your Answer MUST be numbered and take the following format for each ranking:
    - [ACTUAL PAPER TITLE]
    - Reason: [Brief explanation of why this paper is relevant to the query.]

Additionally, DO NOT repeat papers in your ranking.

Finally, you also have the coversation history which shows user prompts and your responses. You ARE THE Assistant.
If papers in the context are also in the conversation history, simply DO NOT talk about them in your response AT ALL.

IMPORTANT:
    - DO NOT repeat papers that are in conversation history EVEN WHEN they are highly relevant.
    - For example, if a user asks for papers similar to something you ranked before, do not include that result in your next ranking.
    - You can tell the user that no other papers in the database are relevant to their query if all papers in conversation history have been listed before.

CONVERSATION HISTORY:
{conversation_history}    

PAPER TITLES AND CONTEXT:
{context}

Query:
{query}
"""


def ask_llm(llm_prompt, llm="mistral:7b"):
    return ollama.generate(
        model=llm, 
        prompt=llm_prompt,
        options = {
            "num_ctx" : 10000
        }

        )['response']

def rewrite_query(follow_up, conversation_history, llm="mistral:7b"):
        

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