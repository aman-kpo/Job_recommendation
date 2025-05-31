
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
import json
import re

# load_dotenv()
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
GCP_SERVICE_ACCOUNT = st.secrets["GCP_SERVICE_ACCOUNT"]
GCP_SERVICE_ACCOUNT = GCP_SERVICE_ACCOUNT.replace("\n", "\\n")
GCP_SERVICE_ACCOUNT = json.loads(GCP_SERVICE_ACCOUNT)


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
    # SERVICE_ACCOUNT_FILE = 'synapse-recruitment-34e7b48899b4.json'
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
    # if not os.path.exists(SERVICE_ACCOUNT_FILE):
    #     st.error(f"Service account file not found at {SERVICE_ACCOUNT_FILE}")
    #     return None
    creds = service_account.Credentials.from_service_account_info(
        GCP_SERVICE_ACCOUNT, scopes=SCOPES
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
