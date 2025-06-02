
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
    industry: str = Field(description="The industry the candidate has experience in.(Tech,Legal,Finance/Accounting,Healthcare,Industrial,Logistics,Telecom,Admin,Other)")


class Job(BaseModel):
    job_title: str = Field(description="The title of the job.")
    company: str = Field(description="The company offering the job.")
    location: str = Field(description="The location of the job.")
    skills: List[str] = Field(description="List of skills required for the job.")
    description: str = Field(description="A brief description of the job.")
    relevance_score: float = Field(description="Relevance score of the job to the candidate's resume.")
    industry: str = Field(description="The industry the job is in.(Tech,Legal,Finance/Accounting,Healthcare,Industrial,Logistics,Telecom,Admin,Other)")



# ——— helper to parse a comma-separated tech stack into a set ———
import re

def clean_text(text: str) -> str:
    """
    Remove HTML tags (<...>) and BBCode-like tags ([...]) from the input text,
    then collapse any repeated whitespace/newlines into single spaces.
    """
    # 1. Remove HTML tags:
    #    `<tag attr="…">`   or   `</tag>`   or   `<br/>`, etc.
    no_html = re.sub(r'<.*?>', '', text)

    # 2. Remove BBCode-like tags:
    #    `[tag attr=...]`   or   `[/tag]`, etc.
    no_bbcode = re.sub(r'\[.*?\]', '', no_html)

    # 3. Collapse any runs of whitespace (spaces, tabs, newlines) into a single space
    cleaned = re.sub(r'\s+', ' ', no_bbcode).strip()
    return cleaned


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
        ws = gc.open_by_key('1VmEXeZtAJ80UEW7xB6_fWLF9XFnz3r8mS307AjBWvIc') \
               .worksheet("Cleaned_job_data")
        data = ws.get_all_values()
        df = pd.DataFrame(data[1:], columns=data[0]).fillna("Unknown")
        # parse Tech Stack into a set for each row
        df["Requirements"] = df["Requirements"].apply(clean_text)
        df["Role_Responsibilities"] = df["Role_Responsibilities"].apply(clean_text)
        df['Industry'] = df['Industry'].replace('VC Tech', 'Tech')
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
        ("system", "Your task is to extract structured data from resumes. In the industry field, use one of the following based on the candidate's past experiences and skills: "
                  "Tech, Legal, Finance/Accounting, Healthcare, Industrial, Logistics, Telecom, Admin, Other."),
        ("human", "Extract: {resume_text}. If missing, return Unknown for each field.")
    ])
    return (prompt | sum_llm).invoke({"resume_text": resume_text})


ef eval_jobs(jobs_df, resume_text):
    # 1. Extract candidate’s structured info (including industry)
    response = structure_resume_data(resume_text)
    candidate_industry = response.industry.strip().lower()

    prompts = {"tech":"""Bachelor’s or higher in CS or related field from:
        MIT, Stanford, CMU, UC Berkeley, Caltech, Harvard, Princeton, UIUC, University of Washington,
        Columbia, Cornell, University of Michigan, UT Austin, University of Toronto, University of Waterloo
        📈 Experience & Career Progression
        3–10 years post-graduation
        Average 2+ years per role (no job hoppers)
        Rapid promotions or scope expansion
        🚀 Startup & VC Experience
        Worked at startups backed by YC, Sequoia, a16z, Greylock, etc.
        Founding Engineer experience = strong bonus
        🚫 Red Flags
        Bootcamp grads only
        IT consulting body shops (TCS, Infosys, Wipro)
        C2C contractors
        Visa-dependent (H1B, TN)
        Big Tech-only without startup exposure""",
        "legal": """JD from Top 20 U.S. Law School:
        Yale, Stanford, Harvard, Columbia, UChicago, NYU, Penn, UC Berkeley, UCLA, Duke, Georgetown,
        Cornell, Michigan, Northwestern, UVA
        📈 Experience
        2–7 years practicing law
        Practice area match (e.g., M&A, IP, Securities)
        2+ years per role preferred
        🚫 Red Flags
        JD from unranked/non-accredited
        In-house or policy roles only
        Job hopping
        Not licensed in job state""",
        "finance/accounting":"""Bachelor’s or Master’s in Accounting or Taxation from:
        University of Illinois Urbana-Champaign, University of Texas at Austin, NYU Stern, USC, UC Berkeley, University of Michigan
        📈 Experience
        3–5 years total post-graduation
        At least 2 years at Big 4 or mid-tier public accounting firm
        Exposure to compliance and tax provision (ASC 740)
        📊 Domain Expertise
        Federal, state, local corporate tax returns
        Entity structures (C-Corp, S-Corp, partnerships)
        Familiar with GoSystem, CCH Axcess, or similar tools
        🚫 Red Flags
        No CPA
        Only industry accounting (e.g., Apple, Google)
        Bootcamp or finance-only backgrounds
        Visa-dependent candidates""",
        "healthcare": """Bachelor's degree in Healthcare Administration, Finance, or similar
        Master’s degree (MHA/MBA) is a bonus
        Credentialed via HFMA, AAHAM, or similar organizations
        📈 Experience
        10–15 years total healthcare RCM experience
        5+ years in leadership roles managing $100M+ revenue cycles
        Oversaw end-to-end billing, coding, collections, denials, payer contracting
        📊 Domain Expertise
        Familiar with Epic, Cerner, or Meditech systems
        Track record of reducing AR days and improving collections
        Deep understanding of Medicare, Medicaid, commercial payers
        🚫 Red Flags
        Only clinic or outpatient billing experience
        No leadership track record
        Gaps in employment or C2C/contractor history""",
        "Others": """Analyze the job details with respect to candidate data and determine if it is a good fit for this candidate, but assigning a relevance score out of 10."""

        }
    # 2. Pre‐filter: only keep jobs whose Industry equals candidate’s industry
    #    (case‐insensitive comparison)
    jobs_df['job_industry_lower'] = jobs_df['Industry'].astype(str).str.strip().str.lower()
    filtered = jobs_df[jobs_df['job_industry_lower'] == candidate_industry].copy()

    if filtered.empty:
        st.warning(f"No jobs found in the '{response.industry}' industry.")
        return pd.DataFrame()

    # 3. Build a combined candidate_text string for LLM evaluation
    candidate_text = (
        f"{response.name} {response.location} "
        f"{', '.join(response.skills)} {response.ideal_jobs} "
        f"{response.yoe} {response.experience} {response.industry}"
    )

    # 4. LLM setup for detailed scoring
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    eval_llm = llm.with_structured_output(Job)
    if candidate_industry=="tech":
        system_prompt = f"""You are an expert recruiter your task to analyze job descriptions based on below criteria \n {prompts['tech']} \n"""
    
    elif candidate_industry=="legal":
        system_prompt = f"""You are an expert recruiter your task to analyze job descriptions based on below criteria \n {prompts['legal']} \n"""
    
    elif candidate_industry=="finance/accounting":
        system_prompt = f"""You are an expert recruiter your task to analyze job descriptions based on below criteria \n {prompts['finance/accounting']}
 \n"""
    else:
        system_prompt = f"""You are an expert recruiter your task to analyze job descriptions based on below criteria \n {prompts['Others']} \n"""
    
    # system_msg = """
    # You are an expert recruiter. Filter by location, experience, and skills,
    # then rate relevance out of 10."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Evaluate Job: {job_text} vs Candidate: {candidate_text} and reccomend if it is a good fit for this candidate, by assigning a relevance score out of 10. ")
    ])
    chain = prompt | eval_llm

    # 5. Prepare columns for iteration
    jobs_for_eval = filtered[["Company_Name", "Job_Title", "Industry", "Job_Location", "Requirements","Company_Blurb"]]
    results = []

    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(jobs_for_eval)

    for i, row in enumerate(jobs_for_eval.itertuples(), start=1):
        # Check “Stop Evaluation” flag
        if not st.session_state.evaluation_running:
            status_text.text("Evaluation halted by user.")
            break

        progress_bar.progress(i / total)
        status_text.text(f"Evaluating job {i}/{total}: {row.Job_Title} at {row.Company_Name}")

        # Build a compact job_text for the LLM
        job_text = " ".join([
            row.Job_Title,
            row.Company_Name,
            row.Job_Location,
            row.Requirements,
            str(row.Industry),
            row.Company_Blurb
        ])

        # Invoke the LLM to get a structured Job response
        eval_job = chain.invoke({
            "job_text": job_text,
            "candidate_text": candidate_text
        })
        if eval_job.relevance_score>=8.8:
            st.markdown(f"{eval_job.job_title} at {eval_job.company} is a good fit with relevance score: {eval_job.relevance_score}")

        results.append({
            "job_title":      eval_job.job_title,
            "company":        eval_job.company,
            "location":       eval_job.location,
            "skills":         eval_job.skills,
            "description":    eval_job.description,
            "relevance_score": eval_job.relevance_score,
            "industry":       eval_job.industry,
        })
        time.sleep(5)  # Simulate processing delay

    progress_bar.empty()
    status_text.empty()

    # 6. Return top 10 by relevance_score (descending)
    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(
            by=["relevance_score"], ascending=False
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
