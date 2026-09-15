from openai import OpenAI
import ollama
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv('api_key'),
    base_url="https://api.groq.com/openai/v1"
)


def generate_llm_prompt(context, query, conversation_history):

    return f"""
You are a research paper recommendation assistant.

Your task is to rank the papers provided in the CURRENT PAPER CONTEXT
according to how relevant they are to the user's CURRENT QUERY.

STRICT RULES:

1. You may ONLY recommend papers whose titles appear in the
   CURRENT PAPER CONTEXT below.

2. NEVER invent, guess, or create a paper title.

3. NEVER recommend a paper that does not appear in the
   CURRENT PAPER CONTEXT.

4. Do not use paper titles from the conversation history as candidates.
   The conversation history is ONLY provided to understand the
   conversation and previous user requests.

5. Consider ALL papers in the current paper context before ranking them.

6. Rank the papers by their relevance to the CURRENT QUERY.

7. Do not claim that a paper discusses something unless that information
   is supported by its title or context.

8. Do NOT repeat a paper in your ranking.

9. If a paper is only weakly related, it may still be included, but explain
   why its relevance is weaker.

10. If there are no papers in the current context that are relevant,
    say so. Do NOT invent additional papers to fill the ranking.

11. If the users Query has nothing to do with asking FOR research papers, say
    you do not support that functionality and are only here for research paper
    purposes.


OUTPUT FORMAT:

1. [ACTUAL PAPER TITLE]
   Reason: [Brief explanation of why this paper is relevant (or not).]

2. [ACTUAL PAPER TITLE]
   Reason: [Brief explanation of why this paper is relevant (or not).]

You MUST continue for ALL other papers in the CURRENT PAPER CONTEXT.
(MAKE output is numbered accordindly too)

IMPORTANT:
Every paper title in your answer MUST exactly match a paper title
provided in the CURRENT PAPER CONTEXT.

CONVERSATION HISTORY:
{conversation_history}

CURRENT PAPER CONTEXT:
{context}

CURRENT USER QUERY:
{query}
"""


def ask_llm(llm_prompt):

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "user", "content": llm_prompt}
        ],
        temperature=0
    )

    print(response.usage)

    return response.choices[0].message.content


def rewrite_query(follow_up, conversation_history):

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

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()