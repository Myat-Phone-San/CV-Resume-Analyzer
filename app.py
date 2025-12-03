import streamlit as st
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai.errors import APIError as GeminiAPIError
from google.genai import types
import re
import numpy as np
import os
import json

# --- API Key Setup (Using st.secrets for secure deployment) ---
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    # Fallback/error message if key is missing
    st.error("GEMINI_API_KEY not found in Streamlit secrets. Please configure it.")
    st.stop()

# --- Session States to store values ---
if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False

if "resume" not in st.session_state:
    st.session_state.resume = ""

if "job_desc" not in st.session_state:
    st.session_state.job_desc = ""

if "ats_score" not in st.session_state:
    st.session_state.ats_score = 0.0

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
def calculate_similarity_bert(text1, text2):
    """Calculates the cosine similarity between two texts using SBERT embeddings."""
    with st.spinner('Loading SBERT Model...'):
        @st.cache_resource
        def load_model():
            # Use a slightly smaller model for faster startup if bandwidth is a concern
            return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
        ats_model = load_model()
    
    embeddings1 = ats_model.encode([text1], convert_to_tensor=False)
    embeddings2 = ats_model.encode([text2], convert_to_tensor=False)
    
    similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
    return float(similarity)


# Function to use the Gemini API and force ONLY the essential fields
def get_report(resume, job_desc, ats_score):
    """Generates the minimalist, essential candidate evaluation report in strict JSON format."""
    try:
        client = genai.Client(api_key=api_key)

        # 1. Define the mandatory MINIMALIST JSON structure (Response Schema)
        json_schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "AI_Requirement_Score_5_0": types.Schema(type=types.Type.NUMBER, description="The mean score of 5 critical requirements from 0.0 to 5.0."),
                # Note: ATS_Match_Score is injected later, but included here for completeness
                "Gap_Point_Texts": types.Schema(
                    type=types.Type.ARRAY,
                    description="The 3 to 5 most critical areas where the candidate is deficient relative to the JD.",
                    items=types.Schema(type=types.Type.STRING)
                ),
            },
            required=["AI_Requirement_Score_5_0", "Gap_Point_Texts"] # ATS_Match_Score is added externally
        )

        # 2. Simplified prompt to focus the model on calculation and minimalism
        prompt=f"""
        Analyze the Candidate Resume against the Job Description. Your task is to calculate the average score of 5 critical requirements (0.0 to 5.0) and identify the top 3-5 critical gaps where the candidate's experience is lacking.
        
        **INSTRUCTION:** Your entire response MUST be a minimalist JSON object matching the provided schema. Do NOT include any other fields or text outside the JSON structure.

        **CANDIDATE RESUME:** {resume}
        ---
        **JOB DESCRIPTION:** {job_desc}
        """
        
        # 3. Call the Gemini API with the JSON configuration
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=json_schema
            )
        )
        
        # Parse the JSON immediately to inject the pre-calculated ATS score
        report_data = json.loads(response.text)
        # Inject the pre-calculated SBERT score into the final JSON structure
        report_data["ATS_Match_Score"] = float(f"{ats_score:.4f}") 
        
        return json.dumps(report_data, indent=4) # Return the modified JSON string
        
    except GeminiAPIError as e:
        st.error(f"Gemini API Error: Could not generate report. Details: {e}")
        return json.dumps({"error": "API Error: Report generation failed.", "details": str(e)})
    except Exception as e:
        st.error(f"An unexpected error occurred during report generation: {e}")
        return json.dumps({"error": "Unexpected Error: Report generation failed.", "details": str(e)})


def extract_scores(text):
    """
    Extracts the AI_Requirement_Score_5_0 from the minimalist JSON report.
    Returns the average score as a single-element list of floats.
    """
    try:
        report_data = json.loads(text)
        llm_score = report_data.get("AI_Requirement_Score_5_0")
        if llm_score is not None:
            return [float(llm_score)]
        return []
        
    except json.JSONDecodeError:
        return []
    except Exception:
        return []

# -----------------------------------------------------
# --- Title and Branding for Employer Tool ---
st.title("🧑‍💼 AI Candidate Match Evaluator (Minimalist Output)")
st.markdown("Provides only the **essential metrics** and **critical gaps** for rapid screening.")

# ---
## 🚀 Streamlit Application Workflow

# Displays Form only if the form is not submitted
if not st.session_state.form_submitted:
    with st.form("evaluation_form"):

        resume_file = st.file_uploader(label="Upload Candidate Resume (PDF)", type="pdf")

        st.session_state.job_desc = st.text_area(
            "Enter the Job Description (JD) for this role:",
            placeholder="E.g., Senior Data Scientist: 5+ years experience, proficiency in Python, AWS, and Deep Learning..."
        )

        submitted = st.form_submit_button("Evaluate Candidate Fit")
        if submitted:

            if st.session_state.job_desc and resume_file:
                
                st.session_state.resume = extract_pdf_text(resume_file)
                
                if "Could not extract text" in st.session_state.resume:
                    st.warning("Could not proceed with analysis.")
                else:
                    # Calculate ATS score immediately 
                    st.session_state.ats_score = calculate_similarity_bert(st.session_state.resume, st.session_state.job_desc)
                    st.session_state.form_submitted = True
                    st.rerun()

            else:
                st.warning("Please upload a **Resume** and provide a **Job Description** to analyze.")


if st.session_state.form_submitted:
    
    score_place = st.info("Step 1/2: Calculating ATS Score and preparing data...")
    
    ats_score = st.session_state.ats_score
    
    score_place.info("Step 2/2: Generating Minimalist Structured Report (Max 30 seconds)...")
    
    # 2. Get the Analysis Report from LLM (Gemini)
    report = get_report(st.session_state.resume, st.session_state.job_desc, ats_score)

    # 3. Extract the final LLM Score from the generated JSON
    report_scores = extract_scores(report) 
    
    if report_scores:
        avg_score = report_scores[0]
        avg_score_display = f"{avg_score:.2f} / 5.0"
    else:
        avg_score = 0.0
        avg_score_display = "N/A"

    score_place.success("Evaluation completed successfully!")
    st.markdown("---")

    # --- Display Scores and Key Gaps ---
    
    st.subheader("📊 Essential Candidate Metrics")
    col1, col2 = st.columns(2, border=True)
    
    with col1:
        st.write("Keywords and Contextual Similarity Score:")
        st.metric(label="ATS Match Score (0.0 to 1.0)", value=f"{ats_score:.4f}")

    with col2:
        st.write("Average Requirement Fulfillment Score:")
        st.metric(label="AI Requirement Score (Avg / 5.0)", value=avg_score_display)

    # Try to display the key gaps prominently and prepare data for tables/downloads
    try:
        parsed_json = json.loads(report)
        gaps = parsed_json.get('Gap_Point_Texts', [])
        
        st.markdown("---")
        st.subheader("Key Deficiencies (Gap Points)")
        
        if gaps:
            for gap in gaps:
                st.markdown(f"* ❌ **{gap}**")
        else:
            st.markdown("* **No critical gaps identified.**")
        
    except json.JSONDecodeError:
        # Fallback if the initial parsing failed
        st.warning("Could not parse the initial report for gap points.")
        gaps = [] # Ensure gaps is defined even on failure
        
    st.markdown("---")

    ## 📄 Structured Report Summary Table (Key-Value Format)
    
    st.subheader("📄 Structured Report Summary Table")
    
    try:
        # Re-parse or use existing parsed_json to ensure the latest data
        parsed_json = json.loads(report)

        # 1. Display the numerical scores table
        table_data_metrics = {
            "Metric": ["AI Requirement Score", "ATS Match Score"],
            "Value": [
                f"{parsed_json.get('AI_Requirement_Score_5_0', 'N/A')}/5.0",
                f"{parsed_json.get('ATS_Match_Score', 'N/A'):.4f}" 
            ]
        }
        st.table(table_data_metrics)

        # 2. Display the Gaps as a list/text since they are a complex field
        st.markdown("### Gap Point Texts")
        gaps = parsed_json.get('Gap_Point_Texts', [])
        if gaps:
            # Convert list of strings to list of dictionaries for st.table
            gap_table_data = [{"Gap Text": gap} for gap in gaps]
            st.table(gap_table_data)
        else:
            st.markdown("*No detailed gap points found in the JSON.*")


    except Exception as e:
        st.error(f"Error creating summary table: {e}")


    st.markdown("---")

    # --- Existing Raw JSON Output ---
    st.subheader("Raw Structured Output (JSON)")
    
    try:
        # Show the raw JSON output for debugging/completeness
        st.code(report, language="json")
    except NameError:
        st.error("Report generation failed and the raw JSON is not available.")

    st.markdown("---")

    # --- Action Buttons (Dual Downloads) ---
    
    st.subheader("Report Actions")
    
    # Prepare data for Word/Text Download (formatted key-value pairs)
    try:
        word_data = f"CANDIDATE ESSENTIAL METRICS\n"
        word_data += f"--------------------------------------------------\n"
        word_data += f"ATS MATCH SCORE: {parsed_json.get('ATS_Match_Score', 'N/A')}\n"
        word_data += f"AI REQUIREMENT SCORE: {parsed_json.get('AI_Requirement_Score_5_0', 'N/A')}/5.0\n\n"
        word_data += f"KEY DEFICIENCIES (GAP POINTS):\n"
        for gap in parsed_json.get('Gap_Point_Texts', []):
            word_data += f"- {gap}\n"
    except Exception:
        word_data = "Error: Could not format the structured report for text download."


    col_json, col_word, col_r, _ = st.columns([1.5, 1.5, 1, 3])
    
    with col_json:
        # JSON Download Button
        st.download_button(
            label="📥 Download JSON File",
            data=report,
            file_name="Candidate_Metrics_Report.json",
            mime="application/json",
        )
    
    with col_word:
        # Text/Word Download Button
        st.download_button(
            label="📄 Download Text Report",
            data=word_data,
            file_name="Candidate_Metrics_Summary.txt",
            mime="text/plain",
            help="This summary is formatted for easy viewing in a text editor or Microsoft Word."
        )

    with col_r:
        # Reset/Rerun Button
        if st.button("Evaluate New Candidate", key="reset_button", icon="🔄"):
            st.session_state.form_submitted = False
            st.session_state.resume = ""
            st.session_state.job_desc = ""
            st.session_state.ats_score = 0.0
            st.rerun()
