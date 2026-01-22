# rag_processor.py

import logging
from pathlib import Path
from typing import List

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredURLLoader,
)

logger = logging.getLogger(__name__)

# Usaremos um modelo de embedding leve e eficiente
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)


def get_vector_store_path(agent_storage_path: Path) -> Path:
    """Retorna o caminho para a pasta do vector store de um agente."""
    return agent_storage_path / "files" / "vector_store"


def load_documents(file_path: Path) -> List:
    """Carrega e divide documentos de diferentes tipos."""
    if file_path.suffix == '.pdf':
        loader = PyPDFLoader(str(file_path))
    elif file_path.suffix == '.txt':
        loader = TextLoader(str(file_path), encoding='utf-8')
    else:
        # Adicione outros loaders conforme necessário (ex: .docx, .csv)
        logger.warning(f"Unsupported file type: {file_path.suffix}. Skipping.")
        return []

    documents = loader.load()
    return text_splitter.split_documents(documents)


async def process_and_index_file(agent_storage_path: Path, file_path: Path):
    """Processa um arquivo, cria embeddings e o adiciona ao vector store do agente."""
    logger.info(f"Indexing file '{file_path.name}' for agent...")
    vector_store_path = get_vector_store_path(agent_storage_path)

    try:
        # 1. Carregar e dividir o documento
        docs = load_documents(file_path)
        if not docs:
            return

        # 2. Carregar o FAISS index existente ou criar um novo
        if vector_store_path.exists():
            db = FAISS.load_local(str(vector_store_path), embeddings, allow_dangerous_deserialization=True)
            db.add_documents(docs)
        else:
            db = FAISS.from_documents(docs, embeddings)

        # 3. Salvar o index atualizado
        db.save_local(str(vector_store_path))
        logger.info(f"Successfully indexed '{file_path.name}'. Vector store updated.")
    except Exception as e:
        logger.error(f"Failed to index file {file_path.name}: {e}", exc_info=True)


def search_in_agent_files(agent_storage_path: Path, query: str, top_k: int = 4) -> List[str]:
    """Busca no vector store de um agente e retorna os trechos mais relevantes."""
    vector_store_path = get_vector_store_path(agent_storage_path)
    if not vector_store_path.exists():
        return ["Nenhum arquivo foi indexado para este agente ainda."]

    try:
        db = FAISS.load_local(str(vector_store_path), embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={"k": top_k})
        results = retriever.invoke(query)

        # Formata os resultados para serem úteis ao LLM
        formatted_results = [
            f"Fonte: {doc.metadata.get('source', 'N/A')}\nConteúdo: {doc.page_content}"
            for doc in results
        ]
        return formatted_results if formatted_results else ["Nenhuma informação relevante encontrada nos arquivos."]
    except Exception as e:
        logger.error(f"Error during RAG search for agent: {e}", exc_info=True)
        return [f"Erro ao pesquisar nos arquivos: {e}"]