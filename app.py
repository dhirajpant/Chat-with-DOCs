#necessary imports
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import docx
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load API key
api_key = st.secrets["GOOGLE_API_KEY"]
if not api_key:
    raise ValueError("API key for Google Generative AI not found.")
genai.configure(api_key=api_key)

# getting text from doc
def get_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    return "\n".join([para.text for para in doc.paragraphs])

def get_text_from_txt(txt_file):
    return txt_file.read().decode('utf-8')

def get_file_text(files):
    text = ""
    for file in files:
        try:
            if file.name.endswith(".pdf"):
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""  # Handle None text cases
            elif file.name.endswith(".docx"):
                text += get_text_from_docx(file)
            elif file.name.endswith(".txt"):
                text += get_text_from_txt(file)
            else:
                st.error(f"Unsupported file type: {file.name}")
        except Exception as e:
            st.error(f"Error reading {file.name}: {e}")
    return text

# Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# QA chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, say, "Answer is not available in the context." Do not provide a wrong answer.
    
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# user input
def user_input(user_question):
    # Check if the user is requesting a call
    if "call me" in user_question.lower() or "contact me" in user_question.lower():
        st.write("It seems like you'd like us to call you. Please provide your contact details.")
        
        # form to collect user information
        with st.form(key='contact_form'):
            name = st.text_input("Name")
            phone = st.text_input("Phone Number")
            email = st.text_input("Email")
            submit_button = st.form_submit_button(label="Submit")

        if submit_button:
            if name and phone and email:
                st.success(f"Thank you {name}, we will call you at {phone}. A confirmation has been sent to {email}.")
                # logic to send an email or store the information
            else:
                st.error("Please fill in all fields.")
    else:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

        st.write("Reply:", response["output_text"])

# Streamlit UI
def main():
    st.set_page_config(page_title="Chat DOCs")
    st.header("Chat with DOCs")
    user_question = st.chat_input("Ask any question from the files")
    if user_question:
        st.chat_message("user").markdown(user_question)
        with st.chat_message("assistant"):
            with st.spinner("Generating answers"):
                user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your files (.pdf, .docx, .txt)", accept_multiple_files=True)
        if pdf_docs and st.button("Submit"):
            with st.spinner("Processing..."):
                raw_text = get_file_text(pdf_docs)
                if raw_text:
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
                else:
                    st.error("No text could be extracted from the uploaded Files.")
        st.write("This is made by using Google Generative AI (Gemini)")

if __name__ == "__main__":
    main()
