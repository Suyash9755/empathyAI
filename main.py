# hi i'm suyash. This is the main app file mainly focus on stramlit interface for users input in web deployment

# Importing the neccessory libraries 
import os
import streamlit as st
from streamlit_option_menu import option_menu
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq

# IMPORTING MY API KEYS
from dotenv import load_dotenv
load_dotenv()

PINECONE_API_KEY    = os.getenv('PINECONE_API_KEY', '')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'empathyai')
GROQ_API_KEY        = os.getenv('GROQ_API_KEY', '')

os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
os.environ['GROQ_API_KEY']     = GROQ_API_KEY

# PAGE CONFIG (Used an Bootsrap Icon suported by streamlit)
st.set_page_config(page_title="Empathy AI Chatbot", layout="wide", page_icon="🤖")

# LOADING MODELS (only loads once)
@st.cache_resource # This command helps model to store in memory so, it doesn't need to restart every time.
def load_models():
    embedding_model = HuggingFaceEmbeddings(
        model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    )
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name = PINECONE_INDEX_NAME,
        embedding  = embedding_model
    )
    llm = ChatGroq(model_name='llama-3.1-8b-instant')
    return vectorstore, llm


# CHAT FUNCTION
def chat(query, vectorstore, llm):
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
    docs      = retriever.invoke(query)
    context   = "\n".join([d.page_content for d in docs])
    message   = f"Context:\n{context}\n\nUser said: {query}\n\nGive a warm empathy response:"
    response  = llm.invoke(message)
    return response.content

    # Extract only the text - nothing else , when i tested their are content and other columns are appearing 
    if hasattr(response, 'content'):
        return response.content
    return str(response)

# SIDEBAR NAVIGATION FOR AN WEB APP 
with st.sidebar:
    selected = option_menu(
        'Empathy AI Menu',
        ['Home', 'Chat with AI', 'About'],
        icons        = ['house', 'chat-dots', 'info-circle'],
        default_index = 1
    )


# HOME PAGE
if selected == 'Home':
    st.title('Welcome to Empathy AI Chatbot 🤖')
    st.write('')
    st.write('This chatbot understands your emotions and responds empathetically.')
    st.write('It is trained on the empathetic_dialogues dataset from HuggingFace.')
    st.write('')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info('💬 Uses RAG to find relevant emotional responses')
    with col2:
        st.info('🧠 Powered by LangChain + Groq LLM')
    with col3:
        st.info('📦 Embeddings stored in Pinecone')


# CHAT PAGE
if selected == 'Chat with AI':
    st.title('🤖 Empathy AI Chatbot')
    st.write('Tell me how you are feeling. I am here to listen.')
    st.write('---')
 
    # This stores chat history during the session
    if 'messages' not in st.session_state:
        st.session_state.messages = []
 
    # Show all previous messages
    for msg in st.session_state.messages:
        if msg['role'] == 'user':
            st.chat_message('user').write(msg['text'])
        else:
            st.chat_message('assistant').write(msg['text'])
 
    # Input box at bottom
    user_input = st.chat_input('Type how you are feeling...')
 
    if user_input:
        # Show user message
        st.chat_message('user').write(user_input)
 
        # Save user message
        st.session_state.messages.append({'role': 'user', 'text': user_input})
 
        # Get AI response
        with st.spinner('Typing.....'):
            try:
                vectorstore, llm = load_models()
                response         = chat(user_input, vectorstore, llm)
 
                # Show AI message
                st.chat_message('assistant').write(response)
 
                # Save AI message
                st.session_state.messages.append({'role': 'assistant', 'text': response})
 
            except Exception as e:
                st.error(f'Something went wrong: {str(e)}')


# ABOUT PAGE
if selected == 'About':
    st.title('About This Project')
    st.write('')
    st.write('**Created by:** Suyash')
    st.write('**Dataset:** empathetic_dialogues (HuggingFace)')
    st.write('**Stack:** LangChain + HuggingFace + Pinecone + Groq LLM')
    st.write('**Approach:** RAG (Retrieval Augmented Generation)')
    st.write('**OpenAI: OpenAI can be also used due to limited attempts in free version i avoided OpenAI')
    st.write('')
    st.write('---')
    st.write('### How it works:')
    st.write('1. Dataset loaded from HuggingFace')
    st.write('2. Embeddings created using sentence-transformers')
    st.write('3. Stored in Pinecone vector database')
    st.write('4. User message finds similar responses from Pinecone')
    st.write('5. Groq LLM generates final empathetic response')
