# 🤖 Empathy AI Chatbot

A simple AI chatbot that understands emotions and responds empathetically.
Built using RAG (Retrieval Augmented Generation) on the empathetic_dialogues dataset.

---

## 🚀 Features

- **Multi-Page Interface:** Built with `streamlit-option-menu` (Home, Chat, About)
- **Empathetic Responses:** Trained on empathetic_dialogues HuggingFace dataset
- **RAG Pipeline:** Retrieves relevant context from Pinecone before generating response
- **Simple UI:** Just type your feeling and get an empathetic response

---

## 🛠️ Tech Stack

- **Language:** Python
- **Web Framework:** Streamlit
- **LLM:** Groq (llama3-8b)
- **Embeddings:** HuggingFace sentence-transformers
- **Vector DB:** Pinecone
- **Framework:** LangChain

---

## 📁 Project Structure

```
├── main.py                          # Streamlit web app
├── empathy_chatbot.ipynb         # Data prep + embedding notebook
├── requirements.txt                 # All dependencies
├── .env                             # API keys (fill before running)
├── documents/
│   └── empathetic_dialogues.csv     # Dataset file
└── README.md
```

---

## 📊 How it Works

1. **Dataset:** empathetic_dialogues loaded from HuggingFace
2. **Embeddings:** sentence-transformers/all-MiniLM-L6-v2 converts text to vectors
3. **Storage:** Vectors stored in Pinecone index
4. **Retrieval:** User message → find top 3 similar dialogues from Pinecone
5. **Generation:** Groq LLM generates empathetic response using retrieved context

---

**Created by: Suyash**
