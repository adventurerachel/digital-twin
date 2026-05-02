# Digital Twin

## 🔍 Overview
Digital version of Rachel Phang - ask questions about my professional experience, send notifications to the real Rachel Phang, or just have a chat!
This github repo hosts the code for the digital twin in app.py and any updates feed through to Hugging Face Spaces, where it is hosted with a gradio interface.
Updates use Github Actions to push to Hugging Face Spaces.

## ⚙️ Architecture
- Model: gpt-4.1-mini
- Interface: Gradio
- Hosting: Hugging Face Spaces

## 🚀 Live Demo
https://huggingface.co/spaces/datarachel/digitaltwin

## 🧠 Key Features
- Personalisation logic (conversation history and context management)
- Memory system (RAG chunking, embeddings, vector stores)
- Decision-making framework (tool-calling)

## 🛠️ Tech Stack
Python, Gradio, chromadb

## 📦 Setup (for devs)
pip install -r requirements.txt
python app/app.py
