from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from db.vectorstore import vector_store

# Configuration
OLLAMA_HOST = "http://192.168.1.102:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:14b"
QUESTION = "explain neural scaling laws with references from the context given"
PROMPT_TEMPLATE = """
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""


def execute_rag_inference(question: str, inference_model: str, ollama_host: str):
    docs_vector_store = vector_store(inference_model, ollama_host, "academic-articles")
    llm = OllamaLLM(base_url=ollama_host, model=inference_model)
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context", "question"]
    )
    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=docs_vector_store.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
    )
    result = qa_chain.invoke({"query": question})
    # logging
    print("\nRetrieved Context:")
    docs = docs_vector_store.similarity_search(question)
    for i, doc in enumerate(docs):
        print(f"\nDocument {i+1}:")
        print(doc.page_content[:500] + "...")
    print("\nQuestion:")
    print(question)
    print("\nAnswer:")
    print(result["result"])


if __name__ == "__main__":
    execute_rag_inference(QUESTION, LLM_MODEL, OLLAMA_HOST)
