from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
import json
from db.vectorstore import vector_store, embedder_for

ANSWERING_PROMPT = """
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""

SUFFICIENCY_PROMPT = """
Given the current document set {documents} and original question {question}, analyze whether we have coverage of:  
1. All named entities mentioned in the question  
2. Temporal scope requirements  
3. Conflicting information across sources  
Output 'sufficient' only if all criteria are met, otherwise list missing elements."""

REFORMULATION_PROMPT = """
Given original question {question} and current context {context}, generate a question that:
* Seek to fill missing info not yet provided in the current context
* Request comparative analysis of conflicting information"""

RERANKING_PROMPT = """
Given the original question {question} and the following context of document chunks,
rerank these chunks by relevance and importance, providing as output only a JSON array
where arr[i] is the number of the chunk that evaluated to be the i-th most important
(0 being max evaluation)"""


def make_chain(
    ollama_host: str,
    model: str,
    prompt_template: str,
    input_variables: list[str],
    temp=0.8,
    top_k=40,
    top_p=0.9,
):
    """Create a LangChain RunnableSequence with Ollama LLM"""
    llm = OllamaLLM(
        base_url=ollama_host, model=model, temperature=temp, top_k=top_k, top_p=top_p
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=input_variables)
    return RunnableSequence(prompt | llm)


def format_chunks(chunks: list, shorten: bool = False, enumerate: bool = False) -> str:
    """Format a list of chunks into text for prompting"""

    formatted_chunks = []
    for i, doc in enumerate(chunks):
        content = doc.page_content
        if shorten:
            content = content[:500] + "..."
        if enumerate:
            content = f"Chunk {i}:\n{content}"

        formatted_chunks.append(content)

    return "\n\n".join(formatted_chunks)


def determine_sufficiency(question: str, chunks: list, ollama_host: str) -> bool:
    """Determine if retrieved documents are sufficient to answer the question"""

    chain = make_chain(
        ollama_host=ollama_host,
        model="llama3",
        prompt_template=SUFFICIENCY_PROMPT,
        input_variables=["documents", "question"],
    )
    documents_text = format_chunks(chunks)
    result = chain.invoke({"documents": documents_text, "question": question})
    print(f"Sufficiency answer {result}")

    return "sufficient" in result.lower()


def query_reformulation(question: str, chunks: list, ollama_host: str) -> str:
    """Reformulate the question based on current context"""

    chain = make_chain(
        ollama_host=ollama_host,
        model="llama3.1",
        prompt_template=REFORMULATION_PROMPT,
        input_variables=["question", "context"],
    )
    context = format_chunks(chunks)
    result = chain.invoke({"question": question, "context": context})

    return result


def rerank_chunks(
    question: str, current_chunks: list, new_chunks: list, ollama_host: str
) -> list:
    """Rerank chunks based on relevance to the original question"""

    all_chunks = current_chunks + new_chunks
    formatted_chunks = format_chunks(all_chunks, enumerate=True)
    chain = make_chain(
        ollama_host=ollama_host,
        model="",
        prompt_template=RERANKING_PROMPT,
        input_variables=["question", "chunks"],
    )

    result = chain.invoke({"question": question, "chunks": formatted_chunks})
    try:
        ranked_indices = json.loads(result.strip())
        if len(ranked_indices) > 25:
            ranked_indices = ranked_indices[:25]

        return [all_chunks[i] for i in ranked_indices]
    except json.JSONDecodeError as e:
        print(f"Failed to parse reranking result: {e}")
        return all_chunks


def final_answer(question: str, chunks: list, inference_model: str, ollama_host: str):
    context = format_chunks(chunks)
    chain = make_chain(
        ollama_host=ollama_host,
        model=inference_model,
        prompt_template=ANSWERING_PROMPT,
        input_variables=["context", "question"],
    )

    result = chain.invoke({"context": context, "question": question})
    print("\nQuestion:")
    print(question)
    print("\nAnswer:")
    print(result)


def retrieve_relevant_chunks(question: str, inference_model: str, ollama_host: str):
    """Retrieve relevant documents from vector store based on question"""

    docs_vector_store = vector_store(inference_model, ollama_host, "academic-articles")
    chunks = docs_vector_store.similarity_search(question)
    return chunks


def execute_rag_inference(question: str, inference_model: str, ollama_host: str):
    """Execute RAG pipeline with retrieved context"""

    chunks = []
    it = 0
    max_iterations = 50

    while it < max_iterations and not determine_sufficiency(
        question, chunks, ollama_host
    ):
        print(f"Starting iteration {it}")
        query = query_reformulation(question, chunks, ollama_host)
        print(f"New query: {query}")
        new_chunks = retrieve_relevant_chunks(query, inference_model, ollama_host)
        print(f"Fetched new {len(new_chunks)} chunks")
        chunks = rerank_chunks(question, chunks, new_chunks, ollama_host)
        print(f"Retained {len(chunks)} chunks after reranking")

        it += 1

    final_answer(question, chunks, inference_model, ollama_host)
