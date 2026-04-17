import os
from dotenv import load_dotenv
from openai import OpenAI
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Initialize OpenAI client for the LLM part (Generation)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# def get_embeddings_model(use_huggingface=True):
#     """
#     Returns the chosen embedding model. 
#     Set use_huggingface=True to run locally and save OpenAI credits.
#     """
#     if use_huggingface:
#         # Runs locally on your CPU/GPU
#         return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     else:
#         # Requires OpenAI API Quota
#         return OpenAIEmbeddings()

# Updated imports for LangChain 0.2+
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embeddings_model(use_huggingface=True):
    if use_huggingface:
        # This will now use the stable 4.38.0 transformers internally
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    else:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings()


# Converts chunks → embeddings using FAISS
def create_embeddings(chunks, use_huggingface=True):
    texts = [c["text"] for c in chunks]
    
    # Initialize the chosen embedding model
    embeddings_model = get_embeddings_model(use_huggingface)
    
    # Create the FAISS vector store
    vector_store = FAISS.from_texts(texts, embeddings_model)
    return vector_store


# def create_embeddings(chunks):
#     texts = [c["text"] for c in chunks]

#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )

#     vector_store = FAISS.from_texts(texts, embeddings)
#     return vector_store

# Retrieves top-k relevant chunks
def retrieve(vector_store, query, k=5):
    # FAISS uses the embedding model attached to it during creation
    docs = vector_store.similarity_search(query, k=k)
    
    context = "\n\n".join([doc.page_content for doc in docs])
    return context

# Generates answer using LLM (GPT-4o-mini)
def generate_answer(query, context):
    prompt = f"""
You must strictly answer using ONLY the provided context.

- Do NOT use outside knowledge
- If unsure, say "I don't know"

Context:
{context}

Question:
{query}

Answer:
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"


















# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from openai import OpenAI
# from dotenv import load_dotenv
# import os
# # pip install sentence-transformers
# from langchain_community.embeddings import HuggingFaceEmbeddings

# # Replace OpenAIEmbeddings with this:
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# # Load environment variables from .env
# load_dotenv()

# # Initialize OpenAI client using API key from environment
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # Converts chunks → embeddings using FAISS
# def create_embeddings(chunks):
#     texts = [c["text"] for c in chunks]
    
#     embeddings = OpenAIEmbeddings()  # requires OPENAI_API_KEY
#     vector_store = FAISS.from_texts(texts, embeddings)

#     return vector_store


# # Retrieves top-k relevant chunks
# def retrieve(vector_store, query, k=5):
#     docs = vector_store.similarity_search(query, k=k)
    
#     context = "\n\n".join([doc.page_content for doc in docs])
#     return context


# # Generates answer using LLM
# def generate_answer(query, context):
#     prompt = f"""
# You must strictly answer using ONLY the provided context.

# - Do NOT use outside knowledge
# - If unsure, say "I don't know"

# Context:
# {context}

# Question:
# {query}

# Answer:
# """

#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0
#         )

#         return response.choices[0].message.content.strip()

#     except Exception as e:
#         return f"Error generating response: {str(e)}"