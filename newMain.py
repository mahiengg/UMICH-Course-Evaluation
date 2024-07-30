import streamlit as st

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain
from langchain_community.embeddings import OpenAIEmbeddings

from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain import HuggingFaceHub, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

from dotenv import load_dotenv


load_dotenv()


os.environ['OPENAI_API_KEY'] = st.secrets.OPENAI.OPENAI_API_KEY
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

# initialize Hub LLM
hub_llm = HuggingFaceHub(
        repo_id='google/flan-t5-xl',
    model_kwargs={'temperature':1e-10}
)


st.set_page_config("Course Completion Checker")
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
        #st.write("getPdfText",text)
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    #embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def DoubleVerifyResult(firstResult,handbook_text,user_transcript,user_question,user_instruction):
    llm = ChatOpenAI(model='gpt-4o',
             temperature=1,)
    
    anotherNewTemplate = """From the firstResult, Your job is to double verify the firstResult according to the user_questions and user_instruction and
    chat_history related to the given two pdf's. You output should always align with the user_instruction and user_question
    
    Transcript: {transcript}
    Handbook Text: {handbook_text}
    FirstResult:{firstResult}
    user_question: {user_question}
    chat_history:{chat_history}
    user_instruction:{user_instruction}

    Answer:
    """ 
    
    prompt = PromptTemplate(template = anotherNewTemplate, input_variables = ["firstResult","handbook_text", "transcript","user_question","chat_history","user_instruction"])
    memory = ConversationBufferMemory(memory_key="chat_history",input_key="firstResult")
    chain = LLMChain(llm = llm, prompt=prompt,verbose=True,memory=memory)
    chat_history= st.session_state['messages']
    finalResult: dict = chain.predict(
    firstResult = firstResult,handbook_text= handbook_text,
      transcript=user_transcript,user_question=user_question,chat_history=chat_history,
      user_instruction=user_instruction)

    st.session_state.messages.append({"role": "assistant", "content": finalResult})



def user_input(user_transcript, handbook_text,user_question, user_instruction):
    #embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    #embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    #new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    #docs = new_db.similarity_search(user_transcript)
    #st.write("docs" , docs)
 
    
    demoTemplate = """
    You are the pdf document reader, Read the handbook_text and  transcript. Analysis and understand  the main theme of the two pdfs. 
    Compare the two given pdf's topics.
    give the result according to the user_question and user_instruction

 
    Transcript: {transcript}
    Handbook Text: {handbook_text}
    user_question: {user_question}
    chat_history:{chat_history}
    Answer:
    """
    prompt = PromptTemplate(template = demoTemplate, input_variables = ["chat_history","user_question","handbook_text", "transcript","user_instruction"])
    memory = ConversationBufferMemory(memory_key="chat_history",input_key="user_question")
    llm = ChatOpenAI(model='gpt-4o',
             temperature=1,)
    
    chain = LLMChain(llm = llm, prompt=prompt,verbose=True,memory = memory)
    inputs = {
                "chat_history": st.session_state['messages'],
                "user_question": user_question,
                "handbook_text": handbook_text,
                "transcript": user_transcript,
                "user_instruction":user_instruction
            }
    prediction_msg: dict = chain.predict(
      **inputs)
    print(prediction_msg)
    DoubleVerifyResult(prediction_msg, handbook_text,user_transcript, user_question,user_instruction)
        # Add user message to chat history
    #st.session_state.messages.append({"role": "assistant", "content": prediction_msg})

  



def compare_course_materials(handbook_docs, transcript_docs, user_question,user_instruction):
     # Extract text from uploaded PDFs
    handbook_text = get_pdf_text(handbook_docs)
    transcript_text = get_pdf_text(transcript_docs)
    # Get text chunks
    handbook_chunks = get_text_chunks(handbook_text)
    #st.write(handbook_chunks)
     # Create vector stores
    get_vector_store(handbook_chunks)
    transcript_chunks = get_text_chunks(transcript_text)
    #st.write(transcript_chunks)
    
    #transcript_store = get_vector_store(transcript_chunks)
    if transcript_text and user_question and user_instruction:
        user_input(transcript_text,handbook_chunks,user_question,user_instruction)
    #return remaining_courses


def main():
   
    st.header("Lets chat with our Chat Bot")
    if "messages" not in st.session_state:
        st.session_state.messages = []
   
    course_handbook_pdf_docs = st.file_uploader("Upload your First PDF File here", accept_multiple_files=True)
    transcript_docs = st.file_uploader("Upload your Second PDF here", accept_multiple_files=True)

    # Initialize session state to store conversation history
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    instruction = """
    1. Compare the given two pdf files and find the main theme
    2. Examine the layout, headings, subheadings, Subsections of both PDF files. 
    3. Identify and compare the primary theme presented in the two PDFs.
    4. Note any differences in numerical data, statistics, charts, or graphs between the two documents
    5. Summarize the content of the two pdfs.
    """

  
    user_instruction = st.text_area("Your Instrutions", placeholder= instruction)
    if not user_instruction:
        user_instruction = instruction
    user_question = st.text_input('Ask your questions', key='user_input')
    st.session_state.messages.append({"role": "user", "content": user_question})
    if st.button("Submit & Process it"):
         if course_handbook_pdf_docs and transcript_docs and user_question and user_instruction:
             with st.spinner("Processing..."):
                compare_course_materials(course_handbook_pdf_docs, transcript_docs, user_question,user_instruction)
                for message in st.session_state.messages:
                    if message["content"].strip():
                        if message["role"] == "user":
                            with st.chat_message(message["role"]):
                                st.markdown(message["content"])
                        else:
                            with st.chat_message(message["role"]):
                                st.markdown(message["content"])
                st.success("Done")
         else:
             st.error('Check all required fields', icon="🚨")
                
    st.markdown(
    """
    <style>
    .st-emotion-cache-1gv3huu {
        background-color: #00274c;
        color:white
    }
    .st-emotion-cache-1gwvy71 h1{
         color:#ffff
    }
        .st-c0 {
    min-height: 195px;
}
    </style>
    """,
    unsafe_allow_html=True
)
    with st.sidebar:
        st.image("https://umdearborn.edu/sites/default/files/styles/header_logo/public/2021-02/umdearborn_horizontal_white_nobg.png?itok=PlxnmbMb", use_column_width=True)
        st.title("Course Completion Checker Chat Bot")
        # pdf_docs = st.file_uploader("Upload your Student handbook PDF File here", accept_multiple_files=True)
        # if st.button("Submit & Process PDF"):
        #     with st.spinner("Processing..."):
        #         raw_text = get_pdf_text(pdf_docs)
        #         st.write("rawtext", raw_text)
        #         text_chunks = get_text_chunks(raw_text)
        #         st.write("text_chunks", text_chunks)
        #         #get_vector_store(text_chunks)
        #         st.success("Done")
        
      



if __name__ == "__main__":
    main()
