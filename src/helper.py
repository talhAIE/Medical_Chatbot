from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings

# load and extract text from pdf files

def load_pdf_data(data_path):
    loader=DirectoryLoader(
        path=data_path,
        glob='*.pdf',
        loader_cls=PyPDFLoader
    ) 
    documents=loader.load()
    return documents



def filter_to_minimal_docs(documents: List[Document]) -> List[Document]:
    '''
    Filters a list of Document objects to only include the page_content and source metadata.'''
    minimal_docs: List[Document] = []
    for doc in documents:
        src=doc.metadata.get('source')
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={'source': src}
            )
        )
    return minimal_docs

# split docs into smaller chunks
def split_docs(documents):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs=text_splitter.split_documents(documents)
    return docs


def download_embeddings():
    embeddings=HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )
    return embeddings
