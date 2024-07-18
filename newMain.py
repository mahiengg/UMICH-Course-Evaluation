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


def DoubleVerifyResult(firstResult,handbook_text,user_transcript):
    llm = ChatOpenAI(model='gpt-4o',
             temperature=1,)
    
    anotherNewTemplate = """From the firstResult, Your job is to recheck and validate once again with handbook_text and transcript using Course code example CIS535.
    First check how many core courses are required and how many are completed and how many are remaining. Then check how many concentration courses are required and
    how many are completed and how many are remaining. 
     
    
    Finally you must give the main one output in table format must with columns for "Course Code," "Course Name," "Concentration Area," and "Status" (Completed/Remaining),suggestion, credits.
    In Status column for completed status add green tick symbol mark, for remaining courses add red cross symbol mark. 
    Add suggestion column and give suggestion for which courses i need to take to complete my degree requirements. Add credits column which tells
    each credits of the couurse and total credits completed currently. 
    Below the Table give suggestion for two lines only if any.
    
    
    Transcript: {transcript}
    Handbook Text: {handbook_text}
    FirstResult:{firstResult}

    Answer:
    """ 
    
    prompt = PromptTemplate(template = anotherNewTemplate, input_variables = ["firstResult","handbook_text", "transcript"])
    chain = LLMChain(llm = llm, prompt=prompt,verbose=True)

    finalResult: dict = chain.run(
    firstResult = firstResult,handbook_text= handbook_text, transcript=user_transcript)
    st.write("final",finalResult)



def user_input(user_transcript, handbook_text,user_question):
    #embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    #embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    #new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    #docs = new_db.similarity_search(user_transcript)
    #st.write("docs" , docs)
 
    
    demoTemplate = """
    You are the academic course evaluator checker. Compare the course code of courses from the given transcript with the provided handbook text, and answer to the user question
    from given {user_question}.Make sure to provide all the details in clear and easy to understand example table format with course name, code, status, credits. 
    Check grading systems, transfer credits, drops and unofficial drops in the handbook_text to evaluate the degree requirements.
 
    Transcript: {transcript}
    Handbook Text: {handbook_text}
    user_question: {user_question}
    chat_history:{chat_history}
    Answer:
    """
    prompt = PromptTemplate(template = demoTemplate, input_variables = ["chat_history","user_question","handbook_text", "transcript"])
    memory = ConversationBufferMemory(memory_key="chat_history",input_key="user_question")
    llm = ChatOpenAI(model='gpt-4o',
             temperature=1,)
    
    chain = LLMChain(llm = llm, prompt=prompt,verbose=True,memory = memory)
    inputs = {
                "chat_history": st.session_state['messages'],
                "user_question": user_question,
                "handbook_text": handbook_text,
                "transcript": user_transcript
            }
    prediction_msg: dict = chain.predict(
      **inputs)
    print(prediction_msg)
    st.write("chat_history", st.session_state['messages'])
        # Add user message to chat history
    st.session_state.messages.append({"role": "assistant", "content": prediction_msg})
    # for message in st.session_state.messages:
    #         if message["role"] == "user" and message["content"] != "":
    #             with st.chat_message(message["role"]):
    #              st.markdown(message["content"])
    #         else:
    #             with st.chat_message(message["role"]):
    #              st.markdown(message["content"])


    #DoubleVerifyResult(prediction_msg, handbook_text, user_transcript)



def compare_course_materials(handbook_docs, transcript_docs, user_question):
     # Extract text from uploaded PDFs
    handbook_text = get_pdf_text(handbook_docs)
    transcript_text = get_pdf_text(transcript_docs)
    # Get text chunks
    handbook_chunks = get_text_chunks(handbook_text)
    handBookSoft = """DEGREE  REQUIREMENTS  
 
COMPU TER AND INFORMAT ION SCIENCE  
 
A candidate for the Ma ster of Science in Computer and Information Science 
(MS- CIS) d egree must hold a Bachelor’s degree with a minimum GPA of 3.0 
on a 4.0 scale from an accred ited institution. The prerequisite courses for 
admission to this program  are as follows: 
 Calculus  I & II* Page | 4  
 CIS 310 (Comput er Organization)*  
 CIS 350 (Data Stru ctures and Al gorithm An alysis)* 
 CIS 450 (Operating Systems)*  
 IMSE 317 (Engineering Prob ability and Statistics)  or lin ear algebr a* 
 
*Prerequisite courses m ay be taken concurrently  within 2 y ears of admiss ion to  program.  
 
Other experience r equired for admission to this program  is as 
follows: 
 Proficiency in at l east 1 hi gh-level programming language, pref erably C/C++ 
1&2 or Java 1&2. 
 
 
Candidates must then complete at least 30 credit hours of graduate 
coursework approved by program  advisor Dr. David Yoon  
(dhyoon@umich.edu ) with a cumulative grade point average of B or better. 
The 30 credit hours of graduate coursework are distributed as follows: 
 
 Project  Option 
o Core  cours es - 9 credit hours  
o Two conce ntration  areas - 12 credit  hours  
o Cognate  courses  - 6 credit hours  
o Project  - 3 credit hours 
 
 Thesis  Option  
    o Core  cours es - 9 credit hours  
    o One concen tration  area - 6 credit  hours 
    o Cognate  courses  - 6 credit hours  
    o CIS electi ve course  - 3 credit  hours  
    o Thesis  - 6 credit hours 
 
1. CORE COURSES (9 credit hours). All students are required to take 
one course from each of the following categories: 
    Category 1 
        o CIS 505  – Algorithm Design and Analysis 
        o CIS 535  – Programmable Mobile/Wireless Technologies 
    and Pervasive Co mputing 
    Category 2 
        o CIS 527  – Computer Networking 
        o CIS 544  – Computer and Network Security 
    Category 3 
        o CIS 574 – Compiler Design 
        o CIS 578 – Advanced Operating Systems Page | 5   
2. CONCENTRATION  AREAS (3 to 12 credit hours) Under the 
Project Option (Area 4 below), students must take four courses from 
two of the concentration areas below. Under the Thesis Option (Area 
5 below), students must take two courses from one concentration area 
and one ele ctive course. 
 Computer Graphics, G eometric Modeling, and Game Design 
    o CIS 515 – Comput er Graphics  
    o CIS 551 - Advanced Co mputer  Graphics 
    o CIS 552 – Information Visuali zation f or Multime dia and G aming 
    o CIS 587 – Comput er Game Design and  Implementation  I 
    o CIS 588 – Comput er Game Design and  Implementation  II 
    o CIS 562 – Information Visuali zation and Comput er Animation 
 Computer Netwo rks and S ecurity 
    o CIS 527* - Comput er Networks 
    o CIS 537 – Advanced Networking and Distribut ed Systems  
    o CIS 544* - Comput er and Network  Security 
    o CIS 546 - Wireless N etwork Se curity and Priv acy 
    o CIS 548 - Security and Priv acy in Cloud Comput ing 
    o CIS 559 - Principles of Social N etwork  Science 
    o CIS 569 - Wireless S ensor Networks 
    o CIS 584 - Advanced Comput er and N etwork Securi ty 
    o CIS 647 - Research Advan ces in N etworking and Distribut ed Systems  
 Data Manag ement and Analyti cs 
    o CIS 534 - The Semantic Web 
    o CIS 536 - Information R etrieval 
    o CIS 548 - Security and Priv acy in Cloud Comput ing 
    o CIS 556 - Database Systems 
    o CIS 5570 - Introd uction to Big Data 
    o CIS 559 - Principles of Social N etwork  Science 
    o CIS 562 - Web Information M anagement  
    o CIS 568 - Data Mining  
    o CIS 5700 - Advanced Data Mining  
    o CIS 584 - Advanced Comput er and N etwork Securi ty 
    o CIS 586 - Advanced Data Management Systems  
    o CIS 658 - Research Advances in Data Management Systems 
 Information Syste ms 
    o CIS 536 - Information R etrieval 
    o CIS 554 - Information Systems An alysis and Design 
    o CIS 555 - Decision Suppo rt and Expert Systems  
    o CIS 556 - Database Systems 
    o CIS 560 - Electronic  Commerce 
    o CIS 564 - Principles of Organizational Information Systems  
    o CIS 571 - Web Services 
    o CIS 572 - Object-Oriented Systems D esign 
    o CIS 579 - Artificial Intelligence Page | 6  
    Software Engi neering 
    o CIS 525 - Web Technolo gy 
    o CIS 535* - Programmable Mobil e/Wireless T echnologies and 
    Pervasive Computing  
    o CIS 553 - Software Engineering 
    o CIS 565 - Software Quality Assu rance 
    o CIS 566 - Software Architecture and Design Patterns 
    o CIS 575 - Software Engineering Management 
    o CIS 577 - Software User  Interface Design     
    o CIS 580 - Software Evolution 
    o CIS 587 - Comput er Game D esign and  Implementation  I 
    o CIS 588 - Comput er Game Design and  Implementation  II 
    o CIS 678 - Advances in Software Engineering Research 
    Syste ms Software 
    o CIS 505* - Algorithm Desi gn and A nalysis 
    o CIS 527* - Comput er Networks 
    o CIS 535* - Programmable Mobil e/Wireless Technologies and 
    Pervasive Computing  
    o CIS 544 - Comput er and Network  Security 
    o CIS 548 - Security and Priv acy in Cloud Comput ing 
    o CIS 569 - Wireless S ensor Networks 
    o CIS 571 - Web Services 
    o CIS 574* - Compil er Design 
    o CIS 578 - Advanced Oper ating Systems  
    o ECE 554 - Emb edded Systems  
 Web Computing  
    o CIS 525 - Web Technolo gy 
    o CIS 534 - The Semantic Web 
    o CIS 535 - Programmable Mobile/ Wireless T echnolo gies and Pervasive 
    Computing  
    o CIS 536 - Information R etrieval 
    o CIS 544 - Comput er and Network  Security 
    o CIS 548 - Security and Priv acy in Cloud Comput ing 
    o CIS 559 - Principles of Social N etwork  Science 
    o CIS 562 - Web Information M anagement  
    o CIS 571 - Web Services 
 
* May not be us ed as bo th core and elective. 
 
3. COGNATE COURSES (6  credit hours) Students can take  any graduate - 
level courses appr oved by the stud ent’s ad visor, as described in the  Rackham 
require ments for graduation.  
4. PROJE CT OPTION (3 cre dit ho urs) Students must take  CIS 6 95 (Master ’s 
Proje ct) for 3  credits. 
5. THESIS O PTION  (6 credit hours) Students must ta ke a CIS elective course 
for 3 credits and CIS 6 99 (Master ’s Thesis) for 6 credits.  
 
 Page | 10  SOFTWARE ENGINEERING  
 
A candidate for the Ma ster of Sc ience in Software Engineering (MS-SWE) 
degree must hold a Bachelor’s degree in computer science and/or computer 
engineering with an overall GPA of 3.0 or higher. The prerequisite courses for 
admission to this program are as follows: 
 Calculus  I & II* 
 CIS 310 (Comput er Organization)*  
 CIS 350 (Data Stru ctures and Al gorithm An alysis)* 
 CIS 450 (Operating Systems)*  
 IMSE 317 (Engineering Probability and Statistics)  or lin ear algebr a* 
 
*Prerequisite courses m ay be taken concurrently  within 2 y ears of admiss ion to program.  
 
 
Other experience r equired for admission to this program  is as follows: 
 Proficiency in calculus, line ar algebra, statisti cs, and physics. 
 
Candidates must then complete at least 30 credit hours of graduate coursework 
approved by program  advisor Dr. Tommy Xu (zwxu@umich.edu ) with a 
cumulative grade point average of B or better. The 30 credit hours of graduate 
coursework are distributed as follows: 
 
 Project  Option 
o Core  cours es - 15 credit hours  
o Applicati on cours es - 9 credit  hours 
o CIS/EC E electiv e course - 3 credit hours  
o Project  - 3 credit hours 
 Thesis  Option  
o Core  cours es - 15 credit hours  
o Applicati on cours es - 9 credit  hours 
o Thesis  - 6 credit hours 
 
 
1. CORE COURSES (15 credit hours) All students are r equired to 
take the following cours es: 
 Three out of the  following five CIS courses: 
            o CIS 553  - Software Engineering  
            o CIS 565 - Software Quality Assurance  
            o CIS 566 - Software Architecture and Design Patterns  
            o CIS 575 - Software Engineering Management  
            o CIS 580 - Data Analytics in Software Engineering  
 ECE 554 – Embedded Systems 
 ECE 574 – Advanced Software Techniques in Engineering Page | 11  concentrations 


2. APP LICATION  COURSES (9 credit hours) Choose  three  courses 
from one  of the  following app lication ar eas: 
 Web Engi neering 
        o CIS 525 – Web Technology 
        o CIS 534 – The Semantic  Web 
        o CIS 536 – Information R etrieval 
        o CIS 559 – Principles of Social N etwork  Science  
        o CIS 562 – Web Information M anagement  
        o CIS 572 – Web Services: Con cepts, Ar chitectures, and Applic ations  
        o CIS 577 – Software User  Interface Design and Analysis 
        o CIS 580 – Software Evolution  
 Game Engi neering 
        o CIS 515 – Comput er Graphics  
        o CIS 552 – Information Visu alization and Mult imedia  Gaming 
        o CIS 577 – Software User  Interface Design and Analysis 
        o CIS 579 – Artificial Intelligence 
        o CIS 580 – Software Evolution  
        o CIS 587 – Game Design and Implement ation I 
        o CIS 588 – Game Design and Implement ation II 
        o ECE 524 – Interactive Media 
        o ECE 5251 – Multim edia Design Tools  I 
        o ECE 5251 – Multim edia Design Tools  II 
 Data Engi neering and  Analyti cs 
        o CIS 556 – Database Systems  
        o CIS 557 0 – Introdu ction to Big Data 
        o CIS 562 – Web Information M anagement  
        o CIS 568 / ECE 537 – Data Mining  
        o CIS 5700 – Advanced Data Mining  
        o CIS 580 – Software Evolution  
        o CIS 586 – Advanced Data Management  
        o ECE 525 – Multim edia Data Stora ge and R etrieval 
 Information and Knowledge Engi neering 
        o CIS 5570 – Introdu ction to Big Data 
        o CIS 559 – Principles of Social N etwork  Science  
        o CIS 5700 – Advanced Data Mining  
        o CIS 579 – Artificial Intelligence 
        o CIS 580 – Software Evolution  
        o ECE 5251 – Multim edia Design Tools  I 
        o ECE 531 – Intelligent Vehicle  Systems  
        o ECE 537 / CIS 568 – Data Mining  
        o ECE 552 – Fuzzy Systems 
        o ECE 576 – Information  Engineering 
        o ECE 577 – Engineering in Virtu al World 
        o ECE 579 – Intelligent Systems  Page | 12  o ECE 583 – Pattern Recognition and Neural Networks 
        o ECE 588 – Robot Visi on 
 Mobile  and Clo ud Co mputing  
        o CIS 535 – Programmable Mobile/ Wireless T echnology and Pervasive 
        Computing  
        o CIS 537 – Advanced Networking and Distribut ed Systems  
        o CIS 546 – Wireless N etwork Security and Priva cy 
        o CIS 548 – Security and Privacy in Cloud Computing  
        o ECE 528 – Cloud Com puting  
        o ECE 535 – Mobile Devices and Ubiquitous Computing  Systems  
 Embedded Syste ms 
        o CIS 535 – Programmable Mobile/ Wireless T echnology and Pervasive 
        Computing  
        o CIS 569 – Wireless S ensor Networks 
        o ECE 505 – Introd uction to Mic roprocessors  and Emb edded Systems  
        o ECE 535 – Mobile Devices and Ubiquitous Computing  Systems  
        o ECE 5541 – Emb edded Networks 
        o ECE 5751 – Reconfigurable Computing  
3. PROJE CT OPTION (3 cre dit ho urs) Students desiring to  obtain  project 
experience are encoura ged to el ect the directed studi es CIS/ECE 591 (3 credits) or 
Proje ct Course CIS ECE 695 (3 credits) to work  under the sup ervision of a fa culty 
advisor. In addition, the  student must ta ke one additional 3-credit course listed abo ve 
or any  CIS/ECE course related to the  project and  appro ved by the Graduate  Program 
Director. 
4. THESIS O PTION  (6 credit hours) Students d esiring to obtain r esearch 
experience are encoura ged to el ect CIS/ECE 699  (6 credits) and work  under the 
supervision of a fa culty advisor. Page | 13  GRADI NG SYST EM 
 """
    #st.write(handbook_chunks)
     # Create vector stores
    get_vector_store(handbook_chunks)
    transcript_chunks = get_text_chunks(transcript_text)
    #st.write(transcript_chunks)
    
    #transcript_store = get_vector_store(transcript_chunks)
    if transcript_text and user_question:
        user_input(transcript_text,handbook_chunks,user_question)
    #return remaining_courses


def main():
   
    st.header("Lets chat with our Chat Bot")
    if "messages" not in st.session_state:
        st.session_state.messages = []
   
    course_handbook_pdf_docs = st.file_uploader("Upload your Course handbook PDF File here", accept_multiple_files=True)
    transcript_docs = st.file_uploader("Upload your Transcript PDF here for check", accept_multiple_files=True)

    # Initialize session state to store conversation history
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    # Display the conversation history

 

    user_question = st.text_input('Ask your questions', key='user_input')
    st.session_state.messages.append({"role": "user", "content": user_question})
    if st.button("Submit & Process it"):
         if course_handbook_pdf_docs and transcript_docs and user_question:
             with st.spinner("Processing..."):
                compare_course_materials(course_handbook_pdf_docs, transcript_docs, user_question)
                # raw_text2 = get_pdf_text(pdf_docs2)
                # text_chunks2 = get_text_chunks(raw_text2)
                # get_vector_store(text_chunks2)
                for message in st.session_state.messages:
                    if message["role"] == "user" and message["content"] != "":
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])
                    else:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])
                st.success("Done")
    # if user_question:
    #     user_input(user_question)

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
