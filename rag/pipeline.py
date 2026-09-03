import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typer import prompt
from rag.retriever import RAGRetriever

load_dotenv()

def create_llm():

    open_router_api = os.getenv("OPENROUTER_API_KEY")

    if not open_router_api:
        raise ValueError("OPENROUTER_API_KEY not found in .env file." )
    llm = ChatOpenAI(
        model="nvidia/nemotron-3-ultra-550b-a55b:free",
        api_key=open_router_api,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.1,
        max_tokens=1024
    )
    return llm

def advanced_rag(query,retriever: RAGRetriever, llm, top_k=8, score_threshold=0.2, return_context=False):
    """
    Retrieve relevant documents and generate
    an answer using the LLM.
    """

    results = retriever.retrieve(query, top_k=top_k, score_threshold=score_threshold)
    if not results:
           return "No relevant documents found for the query."
    context = "\n\n".join([doc["content"] for doc in results])

    source = [{
         'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'Unknown')),
         'page': doc['metadata'].get('page','unknown'),
         'score': doc['similarity_score'],
         'preview': doc['content'][:300]+ '...'
            }for doc in results]
    retrieval_score = max(doc['similarity_score'] for doc in results)
    prompt = f"""
            You are a retrieval-augmented question
            answering assistant.

            Answer the user's query ONLY using the
            information in the context.

            Rules:

            - The context is the source of truth.
            - If the user's assumption is incorrect,
            correct it using the context.
            - Do not say information is missing if the
            context contains it.
            - Do not use outside knowledge.
            - If the context genuinely does not contain
            enough information, say that the provided
            documents do not contain enough information.
            - Answer directly and concisely.

            Context:
            {context}

            Query:
            {query}

            Answer:
            """

    response = llm.invoke([prompt])
    output = {
          'answer': response.content,
          'retrieval_score': retrieval_score,
          'source': source
      }
    if return_context:
          output['context'] = context
    return output
      