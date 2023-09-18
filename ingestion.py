import os
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)


def ingest_docs():
    loader = ReadTheDocsLoader(
        "/home/rameshv/Repos/GenerativeAI/LangChainCourse/langchain-chat-assistant/langchain-docs/api.python.langchain.com/en/latest/index.html"
    )
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )

    documents = text_splitter.split_documents(raw_documents)

    # Prefix https to the source so the user can go to the source directly
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    embeddings = OpenAIEmbeddings()
    print(f"Going to add {len(documents)} to Pinecone")
    Pinecone.from_documents(documents, embeddings, index_name="langchain-doc-index")
    print("****Loading to vectorestore done ***")


if __name__ == "__main__":
    ingest_docs()
