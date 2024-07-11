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
from langchain.chains import SimpleSequentialChain

from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain import HuggingFaceHub, LLMChain

from dotenv import load_dotenv


load_dotenv()


os.environ['OPENAI_API_KEY'] = ${{secrets.OPENAI_API}}.
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'


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
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")






def user_input(user_transcript, handbook_text):
    #embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    #new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    #docs = new_db.similarity_search(user_transcript)
    input_text = "This is the input text."

    llm = ChatOpenAI(model='gpt-4o',
            temperature=1,)
    
    print(output)
    template_s = """You are a Academic course evaluator. Read the given {handbook_text} Handbook_text completely and learn about how many core 
    courses and concentration courses, electives in every majors in the handbook. Memorize it everything.
    I read and learned all  degree requirements needs for every majors in the course handbook {handbook_text}"""

    challenge_template = PromptTemplate(input_variables=["handbook_text"], template=prompt_template)

    challenge_chain = LLMChain(llm=llm, prompt=challenge_template)
        

    #challenge_chain.run("Design a program that calculates the area of a circle when the radius is given as input.")
    #challenge_chain.run(handbook_text)
    #--------------------------------------------------------------------------------------------------------------

    template_p = """As a Academic course evaluator. Read the given transcript and find the major and and it is your job to list out completed core courses
    and concentration area courses from the Transcript for that particular major by comparing with given Handbook_text  and 
    find out the chosen concentration area and find out all the remaining courses are in that particular concentration area and also find out all the remaining core courses.
    Your response should be clear and presice and put it in table format with with columns for "Course Code", "Course Name", "concentration Area" and "Status" (Completed/Remaining).
    Transcript: {transcript}
    Academic course evaluator:  The major is .....
    The  completed core courses of the major is .... 
    and  concentration area courses of the major is ....and the concentration area chosen is... and remaining core courses
    and concentration courses needs to complete are...."""

    solution_template = PromptTemplate(input_variables=["transcript"], template=template_p)
    solution_chain = LLMChain(llm=llm, prompt=solution_template)
    #solution_chain.run(user_transcript)

    full_chain = SimpleSequentialChain(chains=[challenge_chain, solution_chain], 
                                   verbose=True)
    output = full_chain.run(
        transcript= user_transcript
    )
    print(output)
    st.write(output)




def compare_course_materials(handbook_docs, transcript_docs):
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
 Three out of the  followi ng five CIS courses: 
o CIS 553  - Software Engineering  
o CIS 565 - Software Quality Assurance  
o CIS 566 - Software Architecture and Design Patterns  
o CIS 575 - Software Engineering Management  
o CIS 580 - Data Analytics in Software Engineering  
 ECE 554 – Embedded Syste ms 
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
    #get_vector_store(handbook_chunks)
    transcript_chunks = get_text_chunks(transcript_text)
    st.write(transcript_chunks)
    
    #transcript_store = get_vector_store(transcript_chunks)
    if transcript_text:
        user_input(transcript_text,handBookSoft)
    #return remaining_courses




def main():
   
    st.header("Course Completion Checker")

    #user_question = st.text_input("Ask a Question from the PDF Files")
    course_handbook_pdf_docs = st.file_uploader("Upload your Course handbook PDF File here", accept_multiple_files=True)
    transcript_docs = st.file_uploader("Upload your Transcript PDF here for check", accept_multiple_files=True)

    if st.button("Submit & Process it"):
        if course_handbook_pdf_docs and transcript_docs:
            with st.spinner("Processing..."):
                compare_course_materials(course_handbook_pdf_docs, transcript_docs)
                # raw_text2 = get_pdf_text(pdf_docs2)
                # text_chunks2 = get_text_chunks(raw_text2)
                # get_vector_store(text_chunks2)
                st.success("Done")
    # if user_question:
    #     user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
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
