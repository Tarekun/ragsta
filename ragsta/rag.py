from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from db.vectorstore import vector_store, embedder_for

PROMPT_TEMPLATE = """
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""


def determine_sufficiency(question: str, chunks: list) -> bool:
    # recommended small and reasoning focused
    pass


def query_reformulation(question: str, chunks: list) -> bool:
    def verifiy_question_alignment():
        pass

    pass


def reranking(question: str, current_chunks: list, new_chunks: list) -> list:
    pass


def final_answer(question: str, chunks: list, inference_model: str, ollama_host: str):
    print("Retrieved context:")
    for i, doc in enumerate(chunks):
        print(f"\nDocument {i+1}:")
        print(doc.page_content[:500] + "...")
    context = "\n\n".join([doc.page_content for doc in chunks])

    llm = OllamaLLM(base_url=ollama_host, model=inference_model)
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context", "question"]
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    result = chain.invoke({"context": context, "question": question})
    print("\nQuestion:")
    print(question)
    print("\nAnswer:")
    print(result["text"])


def retrieve_relevant_chunks(question: str, inference_model: str, ollama_host: str):
    """Retrieve relevant documents from vector store based on question"""

    docs_vector_store = vector_store(inference_model, ollama_host, "academic-articles")
    chunks = docs_vector_store.similarity_search(question)
    return chunks


def execute_rag_inference(question: str, inference_model: str, ollama_host: str):
    """Execute RAG pipeline with retrieved context"""

    chunks = retrieve_relevant_chunks(question, inference_model, ollama_host)
    final_answer(question, chunks, inference_model, ollama_host)
