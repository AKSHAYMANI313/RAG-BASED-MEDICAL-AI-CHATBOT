from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

#set up LLM (Mistral with Huggingface)

huggingface_repo_id="mistralai/Mistral-7B-Instruct-v0.3"
def load_LLM(hugging_repo_id):
    # Securely set API token using environment variable
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Initialize the Hugging Face Endpoint
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"), 
        temperature=0.5,
        model_kwargs={'max_length': 512} 
)
    return llm
    #Temperature is to maintain the balanace between creativity and accuracy of the retrieved answer. More the value, more the creativity

#Connect LLM with FAISS and create a chain
#Pass a custom prompt- It is a system prompt where a question and the context is first passed and the LLM generates a response
DB_FAISS_PATH="vectorstore/db_faiss"
custom_prompt_template= """
Use the pieces of information provided in the context provided by the user's question,
if you dont know the answer, just say "I dont know". Keep the answer accurate and do not make up an answer
context: {context}
Question: {question}
"""
#Question passed by the user goes to Question, The relevant context retrieved from the DB of vectors will go to the context
def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template,input_variables=["context","question"])
    return prompt

#Re-Load-DB
embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
db=FAISS.load_local(DB_FAISS_PATH, embedding_model,allow_dangerous_deserialization=True)

#QAChain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_LLM(huggingface_repo_id),
    chain_type='stuff', 
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    chain_type_kwargs={'prompt': set_custom_prompt(custom_prompt_template)}
)
#Invoke Query
user_query=input("Write Query:")
response=qa_chain.invoke({'query':user_query})
response = qa_chain.invoke({'query': user_query})
if isinstance(response, dict) and 'result' in response:
    print('Result:', response['result'])
else:
    print('Unexpected response format:', response)