# Chat with DOCs using Google Generative AI (Gemini)

This Streamlit application allows users to upload `.pdf`, `.docx`, and `.txt` files, extracts the text from them, and answers questions based on the content of the documents using Google Generative AI (Gemini) and FAISS for vector search.

## Features

- **Supports multiple file types**: Upload `.pdf`, `.docx`, and `.txt` files.
- **Text extraction**: Extracts text from the uploaded files for processing.
- **Text chunking**: Splits large texts into smaller chunks using LangChain for better embedding.
- **Google Generative AI integration**: Uses Google Generative AI (Gemini) for generating answers to questions.
- **FAISS vector search**: Embeds document chunks into a FAISS index for similarity search and efficient question answering.
- **Interactive Q&A**: Users can ask questions and get answers based on the uploaded document's content.

## Requirements

- Python 3.12 or higher
- The following Python packages:
  - `streamlit`
  - `PyPDF2`
  - `python-docx`
  - `langchain`
  - `langchain_google_genai`
  - `faiss-cpu`
  - `google-generativeai`
  - `python-dotenv`
