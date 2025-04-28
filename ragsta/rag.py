from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain
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


def retrieve_context(question: str, model: str, ollama_host: str):
    """Retrieve relevant documents from vector store based on question"""
    docs_vector_store = vector_store(model, ollama_host, "academic-articles")
    
    # Log retrieved documents for debugging
    print("\nRetrieved Context:")
    docs = docs_vector_store.similarity_search(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    for i, doc in enumerate(docs):
        print(f"\nDocument {i+1}:")
        print(doc.page_content[:500] + "...")
    
    return context


def execute_rag_inference(question: str, inference_model: str, ollama_host: str):
    """Execute RAG pipeline with retrieved context"""
    # First retrieve relevant documents
    context = retrieve_context(question, EMBED_MODEL, ollama_host)
    
    # Then generate answer using the retrieved context
    llm = OllamaLLM(base_url=ollama_host, model=inference_model)
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    
    # Create a simple LLMChain instead of RetrievalQA
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Execute with our retrieved context
    result = chain.invoke({
        "context": context,
        "question": question
    })
    
    print("\nQuestion:")
    print(question)
    print("\nAnswer:")
    print(result["text"])


if __name__ == "__main__":
    execute_rag_inference(QUESTION, LLM_MODEL, OLLAMA_HOST)
