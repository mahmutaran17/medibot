# 🩺 Medical Chatbot with OpenAI & PDF Support

This project is an intelligent **medical assistant chatbot** built using **Python**, **Streamlit**, and **OpenAI's GPT API**. It reads medical PDFs (e.g., cancer research documents), extracts key content, and allows users to ask questions directly about those documents using natural language. It combines **machine learning principles**, **vector similarity search**, and **LLM-based reasoning**.

---

## 🚀 Features

### 🧠 Machine Learning & LLM Integration
- Uses **OpenAI GPT models** (gpt-3.5-turbo, gpt-4) for conversational responses
- Dynamic control over parameters like **temperature**, **evaluation mode**, and **token limits**

### 📄 PDF Extraction & Summarization
- Extracts and chunks content from medical PDFs using libraries like `pdfplumber` or `PyMuPDF`
- Embeds content using `OpenAIEmbeddings` or `HuggingFace` and stores in a **FAISS vector database**
- Performs **contextual question-answering** by matching user queries to relevant PDF chunks

### 💬 Chat Interface
- **Streamlit**-based web UI with:
  - Login / Signup / Demo access
  - User session management
  - PDF upload + Chat input
  - Realtime feedback from OpenAI

### 🧾 Example Use Case

<img width="895" height="458" alt="image" src="https://github.com/user-attachments/assets/beee99d9-3555-4444-83aa-82d4343fdff6" />


> “Upload a lung cancer research paper and ask:  
> *‘What are the treatment options for stage 2B?’*  
> The chatbot retrieves relevant PDF content and answers using GPT.”

---
<img width="888" height="403" alt="image" src="https://github.com/user-attachments/assets/7c9d3ed5-4f95-48d3-9fba-123d57a3888e" />

---

## 🧰 Tech Stack

| Area            | Tool/Library               |
|------------------|----------------------------|
| Programming      | Python                     |
| LLM Integration  | OpenAI API (`openai`)      |
| Vector DB        | FAISS                      |
| PDF Processing   | `pdfplumber`, `PyMuPDF`    |
| Web UI           | Streamlit                  |
| Auth & DB        | SQLite, `streamlit_authenticator` (optional) |
| Embeddings       | `OpenAIEmbeddings`, `tiktoken` |
| Evaluation       | Temp sliders, token usage display |

---

## 🗂️ Project Structure

