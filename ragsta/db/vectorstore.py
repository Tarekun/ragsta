from langchain_ollama import OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector


def embedder_for(model: str):
    if model.startswith("qwen"):
        return "qwen2.5:14b"
    else:
        return "nomic-embed-text"


def vector_store(inference_model: str, ollama_url: str, collection: str) -> PGVector:
    embedder = embedder_for(inference_model)
    embeddings = OllamaEmbeddings(base_url=ollama_url, model=embedder)
    embedding_length = len(embeddings.embed_query("test embedding"))
    print(
        f"computed embedding length for {inference_model} using {embedder}: {embedding_length}"
    )

    return PGVector(
        embeddings=embeddings,
        # TODO: fix this
        connection="postgresql+psycopg://admin:password@localhost:5432/ragsta",
        collection_name=collection,
        create_extension=True,
        embedding_length=embedding_length,
        engine_args={
            "execution_options": {
                "schema_translate_map": {
                    None: embedder,
                },
            },
        },
    )
