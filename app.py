# import streamlit as st
# import pandas as pd
# import PyPDF2
# import io
# import os
# from google.oauth2 import service_account
# import gspread
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import re
# import pandas as pd
# from pydantic import BaseModel, Field
# from typing import List, Set, Dict, Any, Optional # Already have these, but commented for brevity if not all used
# import time # Added for potential small delays if needed
# from langchain_openai import ChatOpenAI
# from langchain_core.messages import HumanMessage # Not directly used in provided snippet
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser # Not directly used in provided snippet
# from langchain_core.prompts import PromptTemplate
# from dotenv import load_dotenv
# load_dotenv()



# OPENAI_API_KEY =os.getenv("OPENAI_API_KEY")

# class structure(BaseModel):
#     name: str = Field(description="Name of the candidate")
#     location: str = Field(description="The location of the candidate.")
#     skills: List[str] = Field(description="List of individual skills of the candidate")
#     ideal_jobs: str = Field(description="List of ideal jobs for the candidate based on past experience.")
#     yoe: str = Field(description="Years of experience of the candidate.")
#     experience: str = Field(description="A brief summary of the candidate's past experience.")

# class Job(BaseModel):
#     job_title: str = Field(description="The title of the job.")
#     company: str = Field(description="The company offering the job.")
#     location: str = Field(description="The location of the job.")
#     skills: List[str] = Field(description="List of skills required for the job.")
#     description: str = Field(description="A brief description of the job.")
#     relevence_score: float = Field(description="Relevance score of the job to the candidate's resume.")


# # Setup Google Sheets connection

# def initialize_google_sheets():
#     try:
#         SERVICE_ACCOUNT_FILE = 'synapse-recruitment-34e7b48899b4.json'
#         SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
        
#         # Verify if the credentials file exists
#         if not os.path.exists(SERVICE_ACCOUNT_FILE):
#             raise FileNotFoundError(f"Service account file not found at {SERVICE_ACCOUNT_FILE}")
        
#         creds = service_account.Credentials.from_service_account_file(
#             SERVICE_ACCOUNT_FILE, 
#             scopes=SCOPES
#         )
#         gc = gspread.authorize(creds)
#         return gc
#     except Exception as e:
#         st.error(f"Failed to initialize Google Sheets: {str(e)}")
#         st.info("Please ensure your service account credentials are correctly set up.")
#         return None

# def load_jobs_data():
#     try:
#         gc = initialize_google_sheets()
#         if gc is None:
#             return None
            
#         job_sheet = gc.open_by_key('1BZlvbtFyiQ9Pgr_lpepDJua1ZeVEqrCLjssNd6OiG9k')
#         job_worksheet = job_sheet.worksheet("paraform_jobs_formatted")
#         job_data = job_worksheet.get_all_values()
        
#         if not job_data:
#             st.warning("No data found in the sheet")
#             return None
            
#         jobs_df = pd.DataFrame(job_data[1:], columns=job_data[0])
#         jobs_df = jobs_df.fillna("Unknown")
#         return jobs_df
#     except gspread.exceptions.APIError as e:
#         st.error(f"Google Sheets API Error: {str(e)}")
#         return None
#     except Exception as e:
#         st.error(f"Error loading jobs data: {str(e)}")
#         return None

# def extract_text_from_pdf(pdf_file):
#     pdf_reader = PyPDF2.PdfReader(pdf_file)
#     text = ""
#     for page in pdf_reader.pages:
#         text += page.extract_text()
#     return text

# def structure_resume_data(resume_text):
#     model_name = "gpt-4o-mini"
#     llm = ChatOpenAI(
#         model=model_name,
#         temperature=0.3,
#         max_tokens=None,
#         timeout=None,
#         max_retries=2,
#     )
#     sum_llm = llm.with_structured_output(structure)
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", "You are a helpful assistant that extracts structured data from resumes."),
#             ("human", "Extract the following information from the resume: {resume_text} if any of the fields are not present, return Unknown as the value."),
#         ]
#     )
#     cat_class = prompt | sum_llm
#     response = cat_class.invoke(
#         {
#             "resume_text": resume_text
#         }
#     )
#     return response


# def eval_jobs(jobs_df, resume_text):
#     response = structure_resume_data(resume_text)
#     candidate_skills = set(skill.lower() for skill in response.skills)
    
#     # Function to check skill matches
#     def get_matching_skills(tech_stack):
#         # Convert string of tech stack to list and clean
#         job_skills = set(skill.strip().lower() for skill in tech_stack.split(','))
#         return len(candidate_skills.intersection(job_skills))

#     # Add matching skills count and filter jobs
#     jobs_df['matching_skills'] = jobs_df['Tech Stack'].apply(get_matching_skills)
#     filtered_jobs = jobs_df[jobs_df['matching_skills'] >= 2].copy()
    
#     if filtered_jobs.empty:
#         st.warning("No jobs found with matching skills requirements")
#         return None

#     candidate_text = f"{response.name} {response.location} {response.skills} {response.ideal_jobs} {response.yoe} {response.experience}"
#     model_name = "gpt-4o-mini"
#     llm = ChatOpenAI(
#         model=model_name,
#         temperature=0.3,
#         max_tokens=None,
#         timeout=None,
#         max_retries=2,
#     )
#     eval_llm = llm.with_structured_output(Job)
#     system = """You are an expert recruiter who evaluates the candidate's profile and reccommends jobs based on it. For recommending follow the below steps:
#     1. Try to consider the sample space as the jobs which match the candidate's location.
#     2. Now based on this sample space try to filter out jobs which expects more years of experience than the candidate has.
#     3. Now in this step try to recommend jobs which match the candidate's skills and experience.
#     4. Now rate the jobs based on how closely they match with the candidate's profile out of 10
#     """
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", system),
#             ("human", "Try to evaluate the Job: {job_text} based on the candidate's profile {candidate_text} and return the job title, company, location, skills, description,  and relevance score."),
#         ]
#     )
#     chain = prompt | eval_llm
#     jobs_df = jobs_df[["Company","Role","One liner","Locations","Tech Stack","Workplace","Industry","YOE"]]

#     progress_bar = st.progress(0)
#     status_text = st.empty()
#     total_jobs = len(filtered_jobs)
#     job_list = []
    
#     st.info(f"Found {total_jobs} jobs with matching skills")
    
#     for i, (_, job) in enumerate(filtered_jobs.iterrows()):
#         progress = (i + 1) / total_jobs
#         progress_bar.progress(progress)
#         status_text.text(f'Evaluating job {i + 1} of {total_jobs}: {job["Role"]} at {job["Company"]}')
        
#         curr_job = {}
#         job_text = f"{job['Role']} {job['Company']} {job['Locations']} {job['Tech Stack']} {job['Workplace']} {job['Industry']} {job['YOE']}"
#         eval_job = chain.invoke(
#             {
#                 "job_text": job_text,
#                 "candidate_text": candidate_text
#             }
#         )
#         curr_job["job_title"] = eval_job.job_title
#         curr_job["company"] = eval_job.company
#         curr_job["location"] = eval_job.location
#         curr_job["skills"] = eval_job.skills
#         curr_job["description"] = eval_job.description
#         curr_job["relevance_score"] = eval_job.relevence_score
#         curr_job["matching_skills"] = job['matching_skills']
        
#         if curr_job["relevance_score"] >= 9:
#             st.markdown(f"🎯 **{curr_job['job_title']}** at {curr_job['company']} in {curr_job['location']} got a score {curr_job['relevance_score']}")
#         job_list.append(curr_job)
    
#     progress_bar.empty()
#     status_text.empty()
    
#     job_list = sorted(job_list, key=lambda x: (x["matching_skills"], x["relevance_score"]), reverse=True)[:10]
#     recc_jobs_df = pd.DataFrame(job_list)
#     return recc_jobs_df 

# def preprocess_text(text):
#     # Convert to lowercase and remove special characters
#     text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
#     return text


# def main():
#     st.title("Resume Evaluator and Job Recommender")
    
#     # Create a session state to track evaluation status
#     if 'evaluation_running' not in st.session_state:
#         st.session_state.evaluation_running = False
#     if 'evaluation_complete' not in st.session_state:
#         st.session_state.evaluation_complete = False
    
#     # File uploader for resume
#     uploaded_file = st.file_uploader("Upload your resume (PDF format)", type=["pdf"])
    
#     # Stop button - only show during evaluation
#     if st.session_state.evaluation_running:
#         if st.button("Stop Evaluation"):
#             st.session_state.evaluation_running = False
#             st.warning("Evaluation stopped by user")
#             return
    
#     if uploaded_file is not None:
#         try:
#             # Only show Generate button if not currently evaluating
#             if not st.session_state.evaluation_running and st.button("Generate Recommendations"):
#                 st.session_state.evaluation_running = True
#                 st.session_state.evaluation_complete = False
                
#                 # Load jobs data first to check authentication
#                 jobs_df = load_jobs_data()
#                 if jobs_df is None:
#                     st.session_state.evaluation_running = False
#                     return
                
#                 resume_text = extract_text_from_pdf(uploaded_file)
#                 if not resume_text.strip():
#                     st.error("The uploaded PDF does not contain any text.")
#                     st.session_state.evaluation_running = False
#                     return
                
#                 resume_text = preprocess_text(resume_text)
#                 st.success("Resume text extracted successfully!")

#                 with st.spinner("Evaluating jobs..."):
#                     recc_jobs_df = eval_jobs(jobs_df, resume_text)
#                     if recc_jobs_df is not None and not recc_jobs_df.empty:
#                         st.write("Recommended Jobs:")
#                         st.dataframe(recc_jobs_df)
#                         st.session_state.evaluation_complete = True
#                     else:
#                         st.warning("No matching jobs found.")
                
#                 st.session_state.evaluation_running = False
                
#             # Show Try Another Resume button after evaluation is complete
#             if st.session_state.evaluation_complete:
#                 if st.button("Try Another Resume"):
#                     st.session_state.evaluation_complete = False
#                     st.rerun()
                        
#         except Exception as e:
#             st.error(f"Error processing the resume: {str(e)}")
#             st.info("Please try again or contact support if the problem persists.")
#             st.session_state.evaluation_running = False
#             return

# if __name__ == "__main__":
#     main()







import streamlit as st
import pandas as pd
import PyPDF2
import os
from google.oauth2 import service_account
import gspread
from pydantic import BaseModel, Field
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import time
from dotenv import load_dotenv
import re

# load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class structure(BaseModel):
    name: str = Field(description="Name of the candidate")
    location: str = Field(description="The location of the candidate.")
    skills: List[str] = Field(description="List of individual skills of the candidate")
    ideal_jobs: str = Field(description="List of ideal jobs for the candidate based on past experience.")
    yoe: str = Field(description="Years of experience of the candidate.")
    experience: str = Field(description="A brief summary of the candidate's past experience.")


class Job(BaseModel):
    job_title: str = Field(description="The title of the job.")
    company: str = Field(description="The company offering the job.")
    location: str = Field(description="The location of the job.")
    skills: List[str] = Field(description="List of skills required for the job.")
    description: str = Field(description="A brief description of the job.")
    relevance_score: float = Field(description="Relevance score of the job to the candidate's resume.")


# ——— helper to parse a comma-separated tech stack into a set ———
def parse_tech_stack(stack):
    if pd.isna(stack) or stack == "" or stack is None:
        return set()
    if isinstance(stack, set):
        return stack
    try:
        if isinstance(stack, str) and stack.startswith("{") and stack.endswith("}"):
            items = stack.strip("{}").split(",")
            return set(item.strip().strip("'\"").lower() for item in items if item.strip())
        return set(s.strip().lower() for s in str(stack).split(",") if s.strip())
    except Exception as e:
        st.error(f"Error parsing tech stack: {e}")
        return set()


def initialize_google_sheets():
    SERVICE_ACCOUNT_FILE = 'synapse-recruitment-34e7b48899b4.json'
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        st.error(f"Service account file not found at {SERVICE_ACCOUNT_FILE}")
        return None
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    return gspread.authorize(creds)


def load_jobs_data():
    gc = initialize_google_sheets()
    if gc is None:
        return None
    try:
        ws = gc.open_by_key('1BZlvbtFyiQ9Pgr_lpepDJua1ZeVEqrCLjssNd6OiG9k') \
               .worksheet("paraform_jobs_formatted")
        data = ws.get_all_values()
        df = pd.DataFrame(data[1:], columns=data[0]).fillna("")
        # parse Tech Stack into a set for each row
        df['parsed_stack'] = df['Tech Stack'].apply(parse_tech_stack)
        return df
    except Exception as e:
        st.error(f"Error loading jobs data: {e}")
        return None


def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    return "".join(page.extract_text() or "" for page in reader.pages)


def structure_resume_data(resume_text):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001",temperature = 0, api_key=GOOGLE_API_KEY)
    sum_llm = llm.with_structured_output(structure)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You extract structured data from resumes."),
        ("human", "Extract: {resume_text}. If missing, return Unknown for each field.")
    ])
    return (prompt | sum_llm).invoke({"resume_text": resume_text})


def eval_jobs(jobs_df, resume_text):
    """
    - Extract structured candidate info
    - Build candidate skill set
    - Pre‐filter jobs by requiring ≥2 overlapping skills
    - For the filtered set, run the LLM‐evaluation loop
    - At each iteration, check st.session_state.evaluation_running;
      if False, break out immediately.
    """
    response = structure_resume_data(resume_text)
    candidate_skills = set(skill.lower() for skill in response.skills)

    # Quick helper to count overlaps
    def matching_skill_count(tech_stack):
        job_skills = set(skill.strip().lower() for skill in tech_stack.split(","))
        return len(candidate_skills & job_skills)

    # Pre‐filter: require ≥2 overlapping skills
    jobs_df['matching_skills'] = jobs_df['Tech Stack'].apply(matching_skill_count)
    filtered = jobs_df[jobs_df['matching_skills'] >= 2].copy()

    if filtered.empty:
        st.warning("No jobs passed the tech‐stack pre‐filter.")
        return pd.DataFrame()

    candidate_text = (
        f"{response.name} {response.location} "
        f"{', '.join(response.skills)} {response.ideal_jobs} "
        f"{response.yoe} {response.experience}"
    )

    # LLM setup
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001",temperature = 0, api_key=GOOGLE_API_KEY)

    eval_llm = llm.with_structured_output(Job)
    system_msg = """
    You are an expert recruiter. Filter by location, experience, and skills, 
    then rate relevance out of 10."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", "Evaluate Job: {job_text} vs Candidate: {candidate_text}.")
    ])
    chain = prompt | eval_llm

    jobs_for_eval = filtered[["Company", "Role", "Locations", "parsed_stack", "YOE", "matching_skills"]]
    results = []

    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(jobs_for_eval)

    for i, row in enumerate(jobs_for_eval.itertuples(), start=1):
        # Check the "Stop Evaluation" flag before each iteration
        if not st.session_state.evaluation_running:
            # User clicked Stop → break out immediately
            status_text.text("Evaluation halted by user.")
            break

        progress_bar.progress(i / total)
        status_text.text(f"Evaluating job {i}/{total}: {row.Role} at {row.Company}")

        job_text = " ".join([
            row.Role,
            row.Company,
            row.Locations,
            ", ".join(row.parsed_stack),
            str(row.YOE)
        ])

        eval_job = chain.invoke({
            "job_text": job_text,
            "candidate_text": candidate_text
        })

        results.append({
            "job_title":      eval_job.job_title,
            "company":        eval_job.company,
            "location":       eval_job.location,
            "skills":         eval_job.skills,
            "description":    eval_job.description,
            "relevance_score": eval_job.relevance_score,
            "matching_skills": row.matching_skills
        })
        time.sleep(5)  # Simulate processing delay

    progress_bar.empty()
    status_text.empty()

    # Build a DataFrame from whatever has been processed so far
    if results:
        df_results = pd.DataFrame(results)
        # Sort by matching_skills first, then relevance_score
        df_results = df_results.sort_values(
            by=["matching_skills", "relevance_score"],
            ascending=[False, False]
        ).head(10)
    else:
        df_results = pd.DataFrame()

    return df_results


def preprocess_text(text):
    return re.sub(r'[^a-zA-Z\s]', '', text.lower())


def main():
    st.title("Resume Evaluator and Job Recommender")

    # Initialize session state flags
    if 'evaluation_running' not in st.session_state:
        st.session_state.evaluation_running = False
    if 'evaluation_complete' not in st.session_state:
        st.session_state.evaluation_complete = False

    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

    # Show “Stop Evaluation” while the loop is running
    if st.session_state.evaluation_running:
        if st.button("Stop Evaluation"):
            # User clicked “Stop” → flip the flag
            st.session_state.evaluation_running = False
            st.warning("User requested to stop evaluation.")

    if uploaded_file is not None:
        # Only show “Generate Recommendations” if not already running
        if (not st.session_state.evaluation_running) and st.button("Generate Recommendations"):
            # Kick off
            st.session_state.evaluation_running = True
            st.session_state.evaluation_complete = False

            # 1. Load jobs
            jobs_df = load_jobs_data()
            if jobs_df is None:
                st.session_state.evaluation_running = False
                return

            # 2. Extract text from PDF
            resume_text = extract_text_from_pdf(uploaded_file)
            if not resume_text.strip():
                st.error("Uploaded PDF contains no text.")
                st.session_state.evaluation_running = False
                return

            resume_text = preprocess_text(resume_text)
            st.success("Resume text extracted successfully!")

            # 3. Run the evaluation (this may take a while)
            with st.spinner("Evaluating jobs…"):
                recs = eval_jobs(jobs_df, resume_text)

            # 4. Display results (or a warning if nothing returned)
            if not recs.empty:
                st.write("Recommended Jobs:")
                st.dataframe(recs)
                st.session_state.evaluation_complete = True
            else:
                st.warning("No matching jobs found or evaluation was halted early.")

            # Mark evaluation as done (or halted)
            st.session_state.evaluation_running = False

        # After evaluation finishes, allow the user to try another resume
        if st.session_state.evaluation_complete:
            if st.button("Try Another Resume"):
                st.session_state.evaluation_complete = False
                st.rerun()


if __name__ == "__main__":
    main()
