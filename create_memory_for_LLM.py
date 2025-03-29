from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader, CSVLoader

#Load Documents
def load_documents(data_path):
    loaders = [
        DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader),
        DirectoryLoader(data_path, glob='*.txt', loader_cls=TextLoader),
        DirectoryLoader(data_path, glob='*.csv', loader_cls=CSVLoader)
    ]
    documents = []
    for loader in loaders:
        documents.extend(loader.load())

    return documents

DATA_PATH = "data/"
document = load_documents(DATA_PATH)



#Create chunks of words
def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    #chunk_overlap is to make sure the context is maintained before and after a specific chunk
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks
text_chunk=create_chunks(document)


#Create Vector embeddings
def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embedding_model

#Store embeddings in vector store to create a knowledge base like FAISS
DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunk,get_embedding_model())
db.save_local(DB_FAISS_PATH)