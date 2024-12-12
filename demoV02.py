import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import base64
import time
import uuid
import io

# Load environment variables
load_dotenv()

# Configuration
load_dotenv()
# Use environment variables for API keys
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

# Page Configuration
st.set_page_config(
    page_title="5MinPaper",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS Styling
st.markdown("""
<style>
    :root {
        --primary-color: #3498db;
        --secondary-color: #2c3e50;
        --background-color: #f4f6f8;
        --text-color: #2c3e50;
        --accent-color: #2ecc71;
    }
    .pdf-viewer {
        width: 100%;
        height: 800px;
        border: 2px solid var(--primary-color);
        border-radius: 10px;
        overflow: auto;
        background-color: white;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    .help-section {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Utility Functions
def pdf_to_base64(pdf_file):
    return base64.b64encode(pdf_file.getvalue()).decode('utf-8')

def is_scanned_pdf(pdf_file):
    try:
        pdf_reader = PdfReader(pdf_file)
        text = "".join([page.extract_text() for page in pdf_reader.pages])
        return len(text.strip()) < 100
    except Exception as e:
        st.error(f"PDF Processing Error: {e}")
        return True

def process_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = "".join([page.extract_text() for page in pdf_reader.pages])

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    text_chunks = text_splitter.split_text(text)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    file_id = str(uuid.uuid4())
    vector_store.save_local(f"faiss_index_{file_id}")

    return file_id

def render_sidebar():
    with st.sidebar:
        st.markdown("<div style='font-size: 1.6rem; font-weight: bold; color: #00b9c6;'>Interact with your Papers</div>", unsafe_allow_html=True)


        # Document Upload Section
        with st.expander("📤 Document Upload", expanded=True):
            pdf_docs = st.file_uploader(
                "Choose PDF",
                type="pdf",
                help="Upload a text-based PDF for analysis"
            )

            if pdf_docs:
                if st.button("Process Document", key="process_doc"):
                    with st.spinner("Processing PDF..."):
                        if is_scanned_pdf(pdf_docs):
                            st.error("Scanned or low-text PDF detected")
                        else:
                            file_id = process_pdf(pdf_docs)
                            st.session_state.pdf_processed = True
                            st.session_state.pdf_file_id = file_id
                            st.session_state.pdf_base64 = pdf_to_base64(pdf_docs)
                            st.session_state.uploaded_pdf = pdf_docs
                            st.success("PDF processed successfully!")

        # PDF Viewing Option
        if hasattr(st.session_state, 'uploaded_pdf'):
            view_pdf = st.checkbox("View PDF", key="view_pdf_checkbox")

        # Help Section
        with st.expander("Help & Support", expanded=False):
            #st.markdown("<div class='help-section'>", unsafe_allow_html=True)
            st.markdown("How to Use 5MinPaper")
            
            help_sections = {
                "Uploading Documents": [
                    "Click on 'Choose PDF' to upload your document",
                    "Ensure the PDF is text-based for best results",
                    "Click 'Process Document' to analyze the file"
                ],
                "Asking Questions": [
                    "Type your question in the input field",
                    "Click 'Get Insights' to receive AI-powered answers",
                    "Questions can range from summarization to specific details"
                ]
            }

            for section, tips in help_sections.items():
                st.markdown(f"#### {section}")
                for tip in tips:
                    st.markdown(f"- {tip}")

            st.markdown("Troubleshooting")
            st.markdown("""
            - **No Text Extraction**: Ensure PDF is not scanned
            - **Slow Response**: Large PDFs might take longer to process
            - **Error Messages**: Check document format and content
            """)
            st.markdown("</div>", unsafe_allow_html=True)

        # About Section
        with st.expander("About Us", expanded=False):
            st.markdown("#### 5MinPaper - Intelligent Scientific Papers Analysis")
            st.markdown("This app was developed by Mr. Oussama SEBROU")
            st.markdown("""
          
            **Version:** 1.0.0
          

            #### Features
            - AI-Powered PDF Analysis
            - Multi-Language Support
            - Context-Aware Question Answering
            - Fast and Precise Insights

           © 2024 5MinPaper Team. All rights reserved.

            [Contact Us](mailto:oussama.sebrou@gmail.com?subject=5MinPaper%20Inquiry&body=Dear%205MinPaper%20Team,%0A%0AWe%20are%20writing%20to%20inquire%20about%20[your%20inquiry]%2C%20specifically%20[details%20of%20your%20inquiry].%0A%0A[Provide%20additional%20context%20and%20details%20here].%0A%0APlease%20let%20us%20know%20if%20you%20require%20any%20further%20information%20from%20our%20end.%0A%0ASincerely,%0A[Your%20Company%20Name]%0A[Your%20Name]%0A[Your%20Title]%0A[Your%20Phone%20Number]%0A[Your%20Email%20Address])
            """)

def display_pdf(pdf_file):
    base64_pdf = base64.b64encode(pdf_file.getvalue()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800px" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def initialize_conversation_history():
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

def update_conversation_history(query, response):
    st.session_state.conversation_history.append({
        'query': query,
        'response': response
    })

def get_conversation_context(max_context_length=3):
    """
    Retrieve conversation context for maintaining continuity
    :param max_context_length: Maximum number of previous interactions to consider
    :return: Formatted context string
    """
    if not hasattr(st.session_state, 'conversation_history'):
        return ""
    
    context_history = st.session_state.conversation_history[-max_context_length:]
    context_str = "\n".join([
        f"Previous Query: {item['query']}\nPrevious Response: {item['response']}"
        for item in context_history
    ])
    
    return context_str

def main():
    # Initialize conversation tracking
    initialize_conversation_history()

    st.markdown("<h1 style='font-size: 3.5rem;'>5MinPaper Chat</h1>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 1.5rem; color: #00b9c6;'>Unlock the Knowledg of your Scientific Paper with Large Language Models (LLMs) AI Technology. Ask insightful questions and receive precise, context-aware answers directly from your documents.</div>", unsafe_allow_html=True)
    st.write("")

    # Render Sidebar
    render_sidebar()

    # Display PDF if View PDF is checked
    if hasattr(st.session_state, 'uploaded_pdf'):
        if st.session_state.view_pdf_checkbox:
            display_pdf(st.session_state.uploaded_pdf)

    # Chat Interface
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)


    # New Chat Button
        # New Chat Button
    if st.button("New Chat", key="new_chat_btn"):
            # Reset session state, preserving initial state
            keys_to_preserve = ['uploaded_pdf', 'pdf_processed', 'pdf_file_id', 'pdf_base64']
            for key in list(st.session_state.keys()):
                if key not in keys_to_preserve:
                    del st.session_state[key]

            # Initialize empty conversation history for new chat
            st.session_state.conversation_history = []
            st.rerun()
    # Display Conversation History
    if hasattr(st.session_state, 'conversation_history'):
        for interaction in st.session_state.conversation_history:
            st.markdown(f"<span style='font-size: 1.5rem; color: #00b9c6;'>**Q:** {interaction['query']}</span>", unsafe_allow_html=True)
            #st.markdown(f"<span style='font-size:24px; color:green;'>**A:** {interaction['response']}</span>", unsafe_allow_html=True)

            #st.markdown(f"**Q:** {interaction['query']}")
            st.markdown(f"**A:** {interaction['response']}")


    def clear_text_area():
        """دالة رد نداء لمسح مربع الإدخال"""
        st.session_state[user_query_key] = ""
    user_query_key = "user_query_input"
    
    user_query = st.text_area(
        "Your Question (Type to expand)",
        placeholder="Ask something about your document...",
        key=user_query_key,
        height=68,
        on_change=clear_text_area )

    if user_query:
        st.write(f"User query:\n{user_query}")
    # لا نقوم بتعديل st.session_state هنا مباشرة

    #user_query_key = "user_query_input" #نضع المفتاح في متغير لسهولة الاستخدام

    #user_query = st.text_area(
     #   "Your Question (Type to expand)",
      #  placeholder="Ask something about your document...",
       # key=user_query_key,
        #height=68  )

    #if user_query:
     #   st.write(f"User query:\n{user_query}")
      #  st.session_state[user_query_key] = ""
       # st.experimental_rerun()

    #user_query = st.text_area(
     #   height=30,
      #  "Your Question",
       # placeholder="Ask something about your document...",
        #key="user_query_input"
    #)

    if st.button("Get Insights", key="insights_btn"):
        if not hasattr(st.session_state, 'pdf_processed') or not st.session_state.pdf_processed:
            st.warning("Please upload and process a PDF first")
            return

        response_placeholder = st.empty()
        with st.spinner("Generating Insights..."):
            try:
                model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.7)

                prompt_template = """
                You are an advanced document analysis and interaction AI your name is forever 5minPaper AI, Your responses are based solely on the provided text.
                Context from Previous Interactions:
                {previous_context}

                Document Context:
                {context}

                User Request:
                {question}

                Advanced Instructions:
                1. Consider previous conversation context
                2. Analyze Document
                3. Provide Precise, Contextual Response
                4. Maintain Conversation Coherence
                6. if the user not ask a translation yet, your answer should be in same language of question wriiten.
                7. Mathematical and Scientific Notation: For any mathematical formulas, scientific symbols, or code snippets, present them like professional LaTeX font formatting for professional and accurate representation.
                 
                """

                prompt = PromptTemplate(
                    template=prompt_template,
                    input_variables=["previous_context", "context", "question"]
                )

                chain = load_qa_chain(
                    model,
                    chain_type="stuff",
                    prompt=prompt
                )

                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

                file_id = st.session_state.get('pdf_file_id', 'default')
                new_db = FAISS.load_local(
                    f"faiss_index_{file_id}",
                    embeddings,
                    allow_dangerous_deserialization=True
                )

                docs = new_db.similarity_search(user_query)

                # Get conversation context
                previous_context = get_conversation_context()

                response = chain(
                    {
                        "previous_context": previous_context,
                        "input_documents": docs, 
                        "question": user_query
                    },
                    return_only_outputs=True
                )

                # Simulate typing response
                displayed_text = ""
                for char in response["output_text"]:
                    displayed_text += char
                    response_placeholder.markdown(displayed_text)
                    time.sleep(0.006)

                # Update conversation history
                update_conversation_history(user_query, response["output_text"])

            except Exception as e:
                st.error(f"Analysis Error: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
