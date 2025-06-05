
import streamlit as st
import pandas as pd
import PyPDF2
import os
from google.oauth2 import service_account
import gspread
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import time
from dotenv import load_dotenv
import json
import re
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from functools import lru_cache

# load_dotenv()
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
GCP_SERVICE_ACCOUNT = st.secrets["GCP_SERVICE_ACCOUNT"]
GCP_SERVICE_ACCOUNT = GCP_SERVICE_ACCOUNT.replace("\n", "\\n")
GCP_SERVICE_ACCOUNT = json.loads(GCP_SERVICE_ACCOUNT)



class structure(BaseModel):
    name: str = Field(description="Name of the candidate")
    location: str = Field(description="The location of the candidate. Extract city and state if possible.")
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
    justification: str = Field(description = "Reason for giving this relevance score and what all areas need to be improved by the candidate")

# ——— Cache expensive operations ———
@lru_cache(maxsize=128)
def get_coordinates_cached(location: str) -> Optional[Tuple[float, float]]:
    """Cached version of get_coordinates to avoid repeated geocoding"""
    try:
        geolocator = Nominatim(user_agent="job_recommender")
        location_obj = geolocator.geocode(location, timeout=10)
        if location_obj:
            return (location_obj.latitude, location_obj.longitude)
    except Exception as e:
        # Don't show warning in cached version to avoid spam
        pass
    return None

@st.cache_data
def load_jobs_data_cached():
    """Cache the jobs data to avoid reloading from Google Sheets"""
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

@st.cache_data
def extract_and_process_resume(_uploaded_file):
    """Cache resume processing to avoid re-processing the same file"""
    reader = PyPDF2.PdfReader(_uploaded_file)
    resume_text = "".join(page.extract_text() or "" for page in reader.pages)
    processed_text = preprocess_text(resume_text)
    return processed_text

# ——— Optimized location filtering functions ———
def clean_location(location: str) -> str:
    """Clean and standardize location string"""
    if pd.isna(location) or location.strip().lower() in ['unknown', '', 'n/a']:
        return ""
    
    # Remove common prefixes/suffixes and clean
    location = re.sub(r'\b(remote|hybrid|on-site|onsite)\b', '', location, flags=re.IGNORECASE)
    location = re.sub(r'[^\w\s,.-|]', '', location)  # Keep pipe separator
    location = re.sub(r'\s+', ' ', location).strip()
    
    return location

def parse_job_locations(job_location: str) -> list:
    """Parse pipe-separated job locations into a list of cities"""
    if not job_location or pd.isna(job_location):
        return []
    
    # Split by pipe and clean each location
    locations = [loc.strip() for loc in job_location.split('|') if loc.strip()]
    
    # Clean each location
    cleaned_locations = []
    for loc in locations:
        cleaned = clean_location(loc)
        if cleaned:
            cleaned_locations.append(cleaned)
    
    return cleaned_locations

def extract_location_from_resume(resume_text: str) -> Optional[str]:
    """Extract location from resume text using common patterns"""
    location_patterns = [
        # City, State patterns
        r'(?:Address|Location|Based in|Live in|Residing in|Located in)[:\s]+([A-Za-z\s]+,\s*[A-Z]{2})',
        r'([A-Za-z\s]+,\s*[A-Z]{2})\s*\d{5}',  # City, State ZIP
        r'([A-Za-z\s]+,\s*[A-Z]{2})(?:\s|$)',   # City, State at end of line
        # International patterns
        r'([A-Za-z\s]+,\s*[A-Za-z\s]+)(?:\s*\d{5,6})?',  # City, Country
    ]
    
    for pattern in location_patterns:
        matches = re.findall(pattern, resume_text, re.IGNORECASE | re.MULTILINE)
        if matches:
            # Return the first reasonable match
            location = matches[0].strip()
            if len(location) > 3 and ',' in location:
                return location
    
    return None

def calculate_distance(loc1_coords: Tuple[float, float], loc2_coords: Tuple[float, float]) -> float:
    """Calculate distance between two coordinates in miles"""
    try:
        return geodesic(loc1_coords, loc2_coords).miles
    except:
        return float('inf')

def is_remote_job(job_location: str) -> bool:
    """Check if job is remote"""
    remote_keywords = ['remote', 'work from home', 'wfh', 'anywhere', 'distributed', 'virtual',"Remote"]
    job_location_lower = job_location.lower()
    return any(keyword in job_location_lower for keyword in remote_keywords)

def check_location_match(job_locations: list, user_location: str) -> Tuple[bool, str]:
    """Check if user location matches any of the job locations (city-level matching)"""
    user_city = user_location.split(',')[0].strip().lower()
    
    for job_loc in job_locations:
        job_city = job_loc.split(',')[0].strip().lower()
        # Direct city name match
        if user_city == job_city:
            return True, job_loc
        # Partial match for common abbreviations
        if user_city in job_city or job_city in user_city:
            return True, job_loc
    
    return False, ""

def filter_jobs_by_location_optimized(jobs_df: pd.DataFrame, user_location: str, max_distance_miles: int = 50, include_exact_matches: bool = True, include_remote: bool = True) -> pd.DataFrame:
    """Optimized version of location filtering with batch coordinate lookup"""
    if jobs_df.empty or not user_location:
        return jobs_df
    
    # Clean user location
    user_location_clean = clean_location(user_location)
    if not user_location_clean:
        return jobs_df
    
    # Get user coordinates (cached)
    user_coords = get_coordinates_cached(user_location_clean)
    if not user_coords:
        st.warning(f"Could not find coordinates for '{user_location_clean}'. Showing all jobs.")
        return jobs_df
    
    # Prepare dataframe for filtering
    jobs_df = jobs_df.copy()
    
    # Vectorized operations where possible
    jobs_df['Is_Remote'] = jobs_df['Job_Location'].apply(is_remote_job)
    jobs_df['Job_Locations_List'] = jobs_df['Job_Location'].apply(parse_job_locations)
    jobs_df['Distance_Miles'] = float('inf')
    jobs_df['Closest_Location'] = ""
    jobs_df['Is_Exact_Match'] = False
    
    # Get unique locations for batch geocoding
    all_job_locations = set()
    for locations_list in jobs_df['Job_Locations_List']:
        all_job_locations.update(locations_list)
    
    # Batch geocode all unique locations
    location_coords_cache = {}
    progress_text = st.empty()
    progress_text.text("Geocoding job locations...")
    
    for location in all_job_locations:
        if location:  # Skip empty locations
            location_coords_cache[location] = get_coordinates_cached(location)
    
    progress_text.empty()
    
    # Process each job with cached coordinates
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_jobs = len(jobs_df)
    exact_matches = 0
    distance_matches = 0
    
    for idx, row in jobs_df.iterrows():
        progress_bar.progress(min(1, (idx + 1) / total_jobs))
        status_text.text(f"Processing job locations...")
        
        if row['Is_Remote']:
            jobs_df.loc[idx, 'Distance_Miles'] = 0
            jobs_df.loc[idx, 'Closest_Location'] = "Remote"
            continue
            
        if not row['Job_Locations_List']:
            continue
        
        # Check for exact city matches first
        if include_exact_matches:
            is_match, matched_location = check_location_match(row['Job_Locations_List'], user_location_clean)
            if is_match:
                jobs_df.loc[idx, 'Distance_Miles'] = 0
                jobs_df.loc[idx, 'Closest_Location'] = matched_location
                jobs_df.loc[idx, 'Is_Exact_Match'] = True
                exact_matches += 1
                continue
        
        # Calculate distance to closest job location using cached coordinates
        min_distance = float('inf')
        closest_location = ""
        
        for location in row['Job_Locations_List']:
            job_coords = location_coords_cache.get(location)
            if job_coords:
                distance = calculate_distance(user_coords, job_coords)
                if distance < min_distance:
                    min_distance = distance
                    closest_location = location
        
        jobs_df.loc[idx, 'Distance_Miles'] = min_distance
        jobs_df.loc[idx, 'Closest_Location'] = closest_location
        
        if min_distance <= max_distance_miles:
            distance_matches += 1
    
    progress_bar.empty()
    status_text.empty()
    
    # Filter jobs that meet criteria
    if include_remote:
        filtered_df = jobs_df[
            (jobs_df['Distance_Miles'] <= max_distance_miles) | 
            (jobs_df['Is_Remote']) |
            (jobs_df['Is_Exact_Match'])
        ]
    else:
        filtered_df = jobs_df[
            ((jobs_df['Distance_Miles'] <= max_distance_miles) | (jobs_df['Is_Exact_Match'])) &
            (~jobs_df['Is_Remote'])
        ]
    
    # Display filtering results
    remote_jobs = len(jobs_df[jobs_df['Is_Remote']])
    included_remote_jobs = len(filtered_df[filtered_df['Is_Remote']])
    
    st.info(f"""
    **Location Filtering Results:**
    - Exact city matches: {exact_matches}
    - Within {max_distance_miles} miles: {distance_matches}
    - Remote jobs: {remote_jobs} (included: {included_remote_jobs})
    - **Total jobs found: {len(filtered_df)}** out of {len(jobs_df)}
    """)
    
    # Add a sample of matched locations for user reference
    if len(filtered_df) > 0:
        sample_locations = filtered_df['Closest_Location'].value_counts().head(5)
        if len(sample_locations) > 0:
            st.write("**Top job locations found:**")
            for location, count in sample_locations.items():
                if location != "Remote":
                    st.write(f"- {location}: {count} jobs")
                elif include_remote:
                    st.write(f"- {location}: {count} jobs")
    
    return filtered_df.drop(['Job_Locations_List', 'Is_Exact_Match'], axis=1)

def get_user_location_preferences(resume_text: str) -> Tuple[str, int, bool, bool]:
    """Get user location and all location preferences"""
    # Initialize session state for location if not exists
    if 'user_location' not in st.session_state:
        st.session_state.user_location = ""
    if 'location_extracted' not in st.session_state:
        st.session_state.location_extracted = False
    if 'location_confirmed' not in st.session_state:
        st.session_state.location_confirmed = False
    if 'location_preferences_set' not in st.session_state:
        st.session_state.location_preferences_set = False
    
    # First try to extract from resume (only once)
    if not st.session_state.location_extracted:
        extracted_location = extract_location_from_resume(resume_text)
        st.session_state.location_extracted = True
        
        if extracted_location:
            st.session_state.extracted_location = extracted_location
            st.success(f"Location detected from resume: {extracted_location}")
        else:
            st.session_state.extracted_location = None
    
    # Show location confirmation if location was extracted
    if hasattr(st.session_state, 'extracted_location') and st.session_state.extracted_location and not st.session_state.location_confirmed:
        confirm = st.radio(
            "Is this location correct?",
            ["Yes", "No, let me enter manually"],
            key="location_confirm"
        )
        
        if confirm == "Yes":
            st.session_state.user_location = st.session_state.extracted_location
            st.session_state.location_confirmed = True
        elif confirm == "No, let me enter manually":
            st.session_state.location_confirmed = True
    
    # Show manual input if no location extracted or user chose manual
    if not st.session_state.user_location or (hasattr(st.session_state, 'extracted_location') and not st.session_state.extracted_location):
        st.info("Please enter your preferred job location:")
        user_location = st.text_input(
            "Location (City, State or City, Country):",
            placeholder="e.g., San Francisco, CA or New York, NY",
            key="manual_location_input",
            value=st.session_state.user_location
        )
        st.session_state.user_location = user_location
    
    # Only show location preferences if user has entered/confirmed a location
    if st.session_state.user_location.strip():
        st.subheader("Job Search Preferences")
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_distance = st.slider(
                "Maximum distance for jobs (miles):",
                min_value=0,
                max_value=200,
                value=50,
                step=10,
                help="Jobs within this distance from your location will be included"
            )
        
        with col2:
            include_exact = st.checkbox(
                "Prioritize exact city matches",
                value=True,
                help="Include jobs in your exact city even if other locations are far"
            )
        
        # Remote jobs checkbox
        include_remote = st.checkbox(
            "Include remote jobs",
            value=True,
            help="Include jobs that can be done remotely"
        )
        
        # Mark preferences as set
        st.session_state.location_preferences_set = True
        
        return st.session_state.user_location, max_distance, include_exact, include_remote
    
    return st.session_state.user_location, 50, True, True

# ——— Original helper functions ———
def clean_text(text: str) -> str:
    """Remove HTML tags and BBCode-like tags from the input text"""
    no_html = re.sub(r'<.*?>', '', text)
    no_bbcode = re.sub(r'\[.*?\]', '', no_html)
    cleaned = re.sub(r'\s+', ' ', no_bbcode).strip()
    return cleaned

def initialize_google_sheets():
    """Initialize Google Sheets connection"""
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
    creds = service_account.Credentials.from_service_account_info(
        GCP_SERVICE_ACCOUNT, scopes=SCOPES
    )
    return gspread.authorize(creds)

@st.cache_data
def structure_resume_data_cached(resume_text):
    """Cached version of resume data structuring"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    sum_llm = llm.with_structured_output(structure)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Your task is to extract structured data from resumes. In the industry field, use one of the following based on the candidate's past experiences and skills: "
                  "Tech, Legal, Finance/Accounting, Healthcare, Industrial, Logistics, Telecom, Admin, Other."),
        ("human", "Extract: {resume_text}. If missing, return Unknown for each field.")
    ])
    return (prompt | sum_llm).invoke({"resume_text": resume_text})

def eval_jobs_with_location_optimized(jobs_df, resume_text, user_location, max_distance, include_exact, include_remote):
    """Optimized version of job evaluation with cached operations"""
    if not user_location:
        st.warning("Please provide a location to get relevant job recommendations.")
        return pd.DataFrame()
    
    # Use cached resume structuring
    response = structure_resume_data_cached(resume_text)
    candidate_industry = response.industry.strip().lower()
    
    # Define prompts (this could also be cached/moved to constants)
    prompts = {
        "tech": """Bachelor's or higher in CS or related field from:
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
        "finance/accounting": """Bachelor's or Master's in Accounting or Taxation from:
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
        Master's degree (MHA/MBA) is a bonus
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
        "others": """Analyze the job details with respect to candidate data and determine if it is a good fit for this candidate, but assigning a relevance score out of 10."""
    }

    # Pre‐filter: only keep jobs whose Industry equals candidate's industry
    jobs_df['job_industry_lower'] = jobs_df['Industry'].astype(str).str.strip().str.lower()
    industry_filtered = jobs_df[jobs_df['job_industry_lower'] == candidate_industry].copy()

    if industry_filtered.empty:
        st.warning(f"No jobs found in the '{response.industry}' industry.")
        return pd.DataFrame()

    # Filter by location using optimized function
    location_filtered = filter_jobs_by_location_optimized(industry_filtered, user_location, max_distance, include_exact, include_remote)
    
    if location_filtered.empty:
        st.warning(f"No jobs found within {max_distance} miles of {user_location} in the '{response.industry}' industry.")
        return pd.DataFrame()

    # Build candidate_text once
    candidate_text = (
        f"{response.name} {response.location} "
        f"{', '.join(response.skills)} {response.ideal_jobs} "
        f"{response.yoe} {response.experience} {response.industry}"
    )

    # LLM setup for detailed scoring (create once, reuse)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    eval_llm = llm.with_structured_output(Job)
    
    # Select appropriate prompt based on industry
    system_prompt = prompts.get(candidate_industry, prompts["others"])
    system_prompt = f"You are an expert recruiter your task to analyze job descriptions based on below criteria \n {system_prompt} \n"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Evaluate Job: {job_text} vs Candidate: {candidate_text} and reccomend if it is a good fit for this candidate, by assigning a relevance score out of 10. ")
    ])
    chain = prompt | eval_llm

    # Prepare jobs for evaluation - select only needed columns once
    jobs_for_eval = location_filtered[["Company_Name", "Job_Title", "Industry", "Job_Location", "Requirements", "Company_Blurb"]]
    results = []

    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(jobs_for_eval)

    # Pre-build job texts to avoid repetitive string operations
    job_texts = []
    for _, row in jobs_for_eval.iterrows():
        job_text = " ".join([
            str(row.Job_Title),
            str(row.Company_Name),
            str(row.Job_Location),
            str(row.Requirements),
            str(row.Industry),
            str(row.Company_Blurb)
        ])
        job_texts.append(job_text)

    for i, (job_text, (_, row)) in enumerate(zip(job_texts, jobs_for_eval.iterrows()), start=1):
        # Check "Stop Evaluation" flag
        if not st.session_state.evaluation_running:
            status_text.text("Evaluation halted by user.")
            break

        progress_bar.progress(i / total)
        status_text.text(f"Evaluating job {i}/{total}: {row.Job_Title} at {row.Company_Name}")

        # Invoke the LLM to get a structured Job response
        eval_job = chain.invoke({
            "job_text": job_text,
            "candidate_text": candidate_text
        })
        
        if eval_job.relevance_score >= 8.8:
            st.markdown(f"{eval_job.job_title} at {eval_job.company} is a good fit with relevance score: {eval_job.relevance_score}")

        results.append({
            "job_title": eval_job.job_title,
            "company": eval_job.company,
            "location": eval_job.location,
            "skills": eval_job.skills,
            "description": eval_job.description,
            "relevance_score": eval_job.relevance_score,
            "industry": eval_job.industry,
            "Reason": eval_job.justification
        })
        time.sleep(5)  # Simulate processing delay

    progress_bar.empty()
    status_text.empty()

    # Return top 10 by relevance_score (descending)
    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(
            by=["relevance_score"], ascending=False
        ).head(10)
    else:
        df_results = pd.DataFrame()

    return df_results

def preprocess_text(text):
    """Clean text for processing"""
    return re.sub(r'[^a-zA-Z\s]', '', text.lower())

def main():
    st.title("Resume Evaluator and Job Recommender")

    # Initialize session state flags
    if 'evaluation_running' not in st.session_state:
        st.session_state.evaluation_running = False
    if 'evaluation_complete' not in st.session_state:
        st.session_state.evaluation_complete = False
    if 'jobs_data_loaded' not in st.session_state:
        st.session_state.jobs_data_loaded = False

    # Load jobs data once at startup
    if not st.session_state.jobs_data_loaded:
        with st.spinner("Loading jobs database..."):
            st.session_state.jobs_df = load_jobs_data_cached()
            st.session_state.jobs_data_loaded = True
            if st.session_state.jobs_df is not None:
                st.success(f"Loaded {len(st.session_state.jobs_df)} jobs from database")

    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

    # Show "Stop Evaluation" while the loop is running
    if st.session_state.evaluation_running:
        if st.button("Stop Evaluation"):
            st.session_state.evaluation_running = False
            st.warning("User requested to stop evaluation.")

    if uploaded_file is not None:
        # Extract resume text using cached function
        resume_text = extract_and_process_resume(uploaded_file)
        
        if resume_text.strip():
            # Show location selection and preferences
            st.subheader("Location Preferences")
            user_location, max_distance, include_exact, include_remote = get_user_location_preferences(resume_text)
            
            # Only show the generate button if we have all required inputs and not currently running
            can_generate = (
                user_location.strip() and 
                st.session_state.get('location_preferences_set', False) and
                not st.session_state.evaluation_running and
                st.session_state.jobs_df is not None
            )
            
            if can_generate:
                if st.button("Generate Recommendations"):
                    st.session_state.evaluation_running = True
                    st.session_state.evaluation_complete = False

                    st.success("Resume text extracted successfully!")

                    # Run the evaluation using cached jobs data
                    with st.spinner("Finding and evaluating relevant jobs..."):
                        recs = eval_jobs_with_location_optimized(
                            st.session_state.jobs_df, 
                            resume_text, 
                            user_location, 
                            max_distance, 
                            include_exact, 
                            include_remote
                        )

                    # Display results
                    if not recs.empty:
                        st.write("Recommended Jobs:")
                        st.dataframe(recs)
                        st.session_state.evaluation_complete = True
                    else:
                        st.warning("No matching jobs found or evaluation was halted early.")

                    # Mark evaluation as done
                    st.session_state.evaluation_running = False
            else:
                if st.session_state.jobs_df is None:
                    st.error("Failed to load jobs database. Please refresh the page.")
                elif not user_location.strip():
                    st.warning("Please enter a location to proceed with job recommendations.")
                elif not st.session_state.get('location_preferences_set', False):
                    st.info("Please set your location preferences above to continue.")

        # After evaluation finishes, allow the user to try another resume
        if st.session_state.evaluation_complete:
            if st.button("Try Another Resume"):
                # Clear location-related session state but keep jobs data
                for key in ['user_location', 'location_extracted', 'location_confirmed', 'extracted_location', 'location_preferences_set']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.evaluation_complete = False
                st.rerun()

if __name__ == "__main__":
    main()




