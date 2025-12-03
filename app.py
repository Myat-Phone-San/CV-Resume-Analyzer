import streamlit as st
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai.errors import APIError as GeminiAPIError
import re
import numpy as np
import os

# --- API Key Setup (Using st.secrets for secure deployment) ---
# NOTE: The GEMINI_API_KEY must be defined in your .streamlit/secrets.toml file.
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    # Fallback/error message if key is missing
    st.error("GEMINI_API_KEY not found in Streamlit secrets.")
    st.stop()

# --- Session States to store values ---
if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False

if "resume" not in st.session_state:
    st.session_state.resume = ""

if "job_desc" not in st.session_state:
    st.session_state.job_desc = ""

# --- Title and Branding for Employer Tool ---
st.title("🧑‍💼 AI Candidate Match Evaluator (Powered by Gemini)")
st.markdown("Instantly assess candidate fit by comparing their resume against your Job Description.")

# -----------------------------------------------------

## ⚙️ Defining Functions

# Function to extract text from PDF
def extract_pdf_text(uploaded_file):
    """Safely extracts text from the uploaded PDF file."""
    try:
        extracted_text = extract_text(uploaded_file)
        return extracted_text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return "Could not extract text from the PDF file."


# Function to calculate similarity (ATS Score)
# Uses st.cache_resource to load the large model only once
def calculate_similarity_bert(text1, text2):
    """Calculates the cosine similarity between two texts using SBERT embeddings."""
    with st.spinner('Loading SBERT Model...'):
        @st.cache_resource
        def load_model():
            return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            
        ats_model = load_model()
    
    embeddings1 = ats_model.encode([text1], convert_to_tensor=False)
    embeddings2 = ats_model.encode([text2], convert_to_tensor=False)
    
    similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
    return float(similarity)


# Rewritten function to use the Gemini API
def get_report(resume, job_desc):
    """Generates a detailed candidate evaluation report using the Gemini LLM."""
    try:
        # Initialize the Gemini Client
        client = genai.Client(api_key=api_key)

        # --- REVISED PROMPT FOR EMPLOYER/RECRUITER PERSPECTIVE ---
        prompt=f"""
        # Role: AI Candidate Analyst

        # Objective:
        - Analyze the provided **Candidate Resume** against the **Job Description (JD)**.
        - Focus on quantifying the candidate's qualification level and identifying key gaps for the employer.

        # Instructions for Analysis:
        1.  **Deconstruct the JD:** Identify 5-7 most critical requirements (e.g., specific skills, years of experience, domain knowledge, education).
        2.  **Evaluate Each Requirement:** For each requirement, assign a score out of 5 (e.g., 4.5/5) based on the evidence in the resume.
        3.  **Scoring Criteria:**
            - **5/5 (✅ Strong Match):** Explicitly mentioned, demonstrated with quantifiable results/experience.
            - **3-4/5 (✅ Good Match):** Mentioned but lacks detail, or indirect evidence is present.
            - **1-2/5 (❌ Weak Match):** Not explicitly mentioned, or only generic keywords are found.
            - **0/5 (❌ No Match):** Requirement is critical, but no related evidence is found in the resume.
            - **(⚠️ Unverifiable):** Use this for requirements that cannot be definitively proven (e.g., "Must be a team player") and provide a reason.
        4.  **Final Section:** The final heading should be "Hiring Recommendation & Key Gaps:" and provide a concise summary of the candidate's suitability and list the top 3-5 critical areas (gaps) where the candidate's resume is deficient relative to the JD.

        # Inputs:
        Candidate Resume: {resume}
        ---
        Job Description: {job_desc}

        # Output Format:
        - Begin each point with the score (e.g., "5/5 ✅ Required Skill: Python Proficiency...").
        - Provide a detailed explanation justifying the score and match symbol.
        """
        # -------------------------------------------------------------

        # Use gemini-2.5-flash for fast and effective reasoning
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text
        
    except GeminiAPIError as e:
        st.error(f"Gemini API Error: Could not generate report. Please check your API key and permissions. Details: {e}")
        return "API Error: Report generation failed."
    except Exception as e:
        st.error(f"An unexpected error occurred during report generation: {e}")
        return "Unexpected Error: Report generation failed."


def extract_scores(text):
    """Extracts all scores in the format X/5 from the LLM report."""
    # Pattern to find scores in the format x/5, where x can be an integer or a float
    pattern = r'(\d+(?:\.\d+)?)/5'
    matches = re.findall(pattern, text)
    # Convert matches to floats
    scores = [float(match) for match in matches]
    return scores

# -----------------------------------------------------
# ---
# ## 🚀 Streamlit Application Workflow

# Displays Form only if the form is not submitted
if not st.session_state.form_submitted:
    with st.form("evaluation_form"):

        # Taking input a Resume (PDF) file
        resume_file = st.file_uploader(label="Upload Candidate Resume (PDF)", type="pdf")

        # Taking input Job Description
        st.session_state.job_desc = st.text_area(
            "Enter the Job Description (JD) for this role:",
            placeholder="E.g., Senior Data Scientist: 5+ years experience, proficiency in Python, AWS, and Deep Learning..."
        )

        # Form Submission Button
        submitted = st.form_submit_button("Evaluate Candidate Fit")
        if submitted:

            # Allow only if Both Resume and Job Description are Submitted
            if st.session_state.job_desc and resume_file:
                
                # Perform extraction
                st.session_state.resume = extract_pdf_text(resume_file)
                
                if st.session_state.resume == "Could not extract text from the PDF file.":
                    st.warning("Could not proceed with analysis.")
                else:
                    st.session_state.form_submitted = True
                    st.rerun()  # Refresh the page to close the form and display results

            # Do not allow if not uploaded
            else:
                st.warning("Please upload a **Resume** and provide a **Job Description** to analyze.")


if st.session_state.form_submitted:
    
    # Placeholders for dynamic updates
    score_place = st.info("Step 1/2: Calculating ATS Similarity Score...")
    
    # 1. Calculate the ATS Score
    ats_score = calculate_similarity_bert(st.session_state.resume, st.session_state.job_desc)
    
    score_place.info("Step 2/2: Generating Detailed Analysis Report (This may take up to 30 seconds)...")
    
    # 2. Get the Analysis Report from LLM (Gemini)
    report = get_report(st.session_state.resume, st.session_state.job_desc)

    # 3. Calculate the Average Score from the LLM Report
    report_scores = extract_scores(report) 
    
    # Correctly calculate the average score out of 5
    if report_scores:
        avg_score = np.mean(report_scores) 
        avg_score_display = f"{avg_score:.2f} / 5.0"
    else:
        avg_score_display = "N/A"

    score_place.success("Evaluation completed successfully!")
    st.markdown("---")

    # --- Display Scores ---
    
    st.subheader("📊 Candidate Fit Overview")
    col1, col2 = st.columns(2, border=True)
    
    with col1:
        st.write("Keywords and Contextual Similarity Score (ATS Perspective):")
        st.metric(label="ATS Match Score (0.0 to 1.0)", value=f"{ats_score:.4f}")
        st.caption("Represents the semantic match between resume and JD. Used for initial filtering.")

    with col2:
        st.write("Average Requirement Fulfillment Score (AI Analyst Perspective):")
        st.metric(label="AI Requirement Score", value=avg_score_display)
        st.caption("Average of individual requirement scores (out of 5) from the detailed report.")
    
    st.markdown("---")

    # --- Display Detailed Report ---
    
    st.subheader("AI Analyst's Detailed Evaluation Report")
    
    st.markdown("---")
    st.markdown(report)
    st.markdown("---")

    # --- Action Buttons ---
    
    st.subheader("Report Actions")
    
    col_d, col_r, _ = st.columns([1, 1, 4])
    
    with col_d:
        # Download Button
        st.download_button(
            label="Download Report (TXT)",
            data=report,
            file_name="Candidate_Evaluation_Report.txt",
            mime="text/plain",
            icon="📥",
        )
    
    with col_r:
        # Reset/Rerun Button
        if st.button("Evaluate New Candidate", key="reset_button", icon="🔄"):
            st.session_state.form_submitted = False
            st.session_state.resume = ""
            st.session_state.job_desc = ""
            st.rerun()

# -----------------------------------------------------
