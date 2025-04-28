from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_postgres.vectorstores import PGVector
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from db.constants import get_engine

# Configuration
OLLAMA_HOST = "http://192.168.1.102:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:14b"
DB_USER = "admin"
DB_PASSWORD = "password"
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "ragsta"
TABLE_NAME = '"article_nomic-embed-text"'

# Connection string
connection_string = (
    f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# Initialize embeddings
embeddings = OllamaEmbeddings(base_url=OLLAMA_HOST, model=EMBED_MODEL)
vector_store = PGVector(
    embeddings=embeddings,
    connection=connection_string,
    collection_name="academic-articles",
    create_extension=True,
    embedding_length=768,
    # 5 months and counting, lets see how long it takes them...
    # https://github.com/langchain-ai/langchain-postgres/pull/138
    engine_args={
        "execution_options": {
            "schema_translate_map": {
                None: "nomic-embed-text",
            },
        },
    },
)

# Initialize LLM
llm = Ollama(base_url=OLLAMA_HOST, model=LLM_MODEL)

# Create a prompt template that includes references
PROMPT_TEMPLATE = """
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE, input_variables=["context", "question"]
)
print("prompt creato")

# Create the QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
)
print("qa chain fatta")

QUESTION = "explain neural scaling laws with references from the context given"

if __name__ == "__main__":
    # Execute the RAG lookup
    result = qa_chain.invoke({"query": QUESTION})

    print("\nQuestion:")
    print(QUESTION)
    print("\nAnswer:")
    print(result["result"])
