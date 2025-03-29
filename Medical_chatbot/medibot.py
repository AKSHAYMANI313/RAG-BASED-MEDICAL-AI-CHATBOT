import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint


DB_FAISS_PATH = 'vectorstore/db_faiss'

# Load FAISS Vectorstore
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# Function to set custom prompt template
def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Hugging Face model details
huggingface_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

# Load LLM from Hugging Face
def load_LLM():
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"), 
        temperature=0.5,
        model_kwargs={'max_length': 512}
    )
    return llm

# Load Vectorstore
try:
    vectorstore = get_vectorstore()
    if vectorstore is None:
        st.error('Failed to load the Vectorstore')
except Exception as e:
    st.error(f"Error loading vectorstore: {e}")

# Streamlit App
def main():
    from langchain.memory import ConversationBufferMemory
    st.title("Ask our Medical Chat-Bot : MEDIBOT")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # User input
    prompt = st.chat_input("Enter your Question here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # Define Custom Prompt
        custom_prompt_template = """
        Use the pieces of information provided in the context to answer the user's question.
        If you don't know the answer, just say "I don't know." Keep the answer accurate and do not make up information.
        Context: {context}
        Question: {question}
        """

        # Load LLM
        llm = load_LLM()

        # Setup RetrievalQA Chain
        
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={'prompt': set_custom_prompt(custom_prompt_template)},  
            memory=memory
        )


        # Get Response from LLM
        response = qa_chain.invoke({'query': prompt})

        # Display response
        st.chat_message('assistant').markdown(response.get('result', "I am Medibot."))  
        st.session_state.messages.append({'role': 'assistant', 'content': response.get('result', "I am Medibot.")})

# Ensure this runs only when the script is executed directly
if __name__ == '__main__':
    main()

