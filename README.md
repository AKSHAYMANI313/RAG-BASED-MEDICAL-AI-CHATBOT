# RAG-BASED-MEDICAL-AI-CHATBOT
Information retrieval and generative AI are combined in a Retrieval-Augmented 
Generation (RAG) based Medical AI chatbot to generate precise and contextually aware 
responses. RAG first obtains information from an external vector database (like FAISS) 
and adds the collected knowledge to the user inquiry before sending it to the LLM (like 
HuggingFace based Mistral-7B in our case), in contrast to standard chatbots that just 
employ pre-trained language models (LLMs). By basing answers on actual data rather 
than the model's previously learnt information, this method increases accuracy in 
answers and makes sure the responses are uptodate . A user inquiry starts the 
chatbot's process, which uses semantic similarity search in FAISS to retrieve the top K 
documents/ text fragments that are saved as vector embeddings. The user's query and 
the documents that were retrieved are then entered into the LLM, which produces a 
response that are more inclined towards the context that the user provides   
