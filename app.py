import streamlit as st
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai.errors import APIError as GeminiAPIError
from google.genai import types # Import types for Response Schema
import re
import numpy as np
import os
import json # Import for handling JSON output

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
st.title("🧑‍💼 AI Candidate Match Evaluator (JSON Output)")
st.markdown("Instantly assess candidate fit by comparing their resume against your Job Description for structured, concise data.")

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


# Rewritten function to use the Gemini API and force JSON output
def get_report(resume, job_desc):
    """Generates a detailed candidate evaluation report using the Gemini LLM in strict JSON format."""
    try:
        client = genai.Client(api_key=api_key)

        # 1. Define the mandatory JSON structure (Response Schema)
        json_schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "Key_Requirements_Evaluation": types.Schema(
                    type=types.Type.ARRAY,
                    description="An array of 5 to 7 most critical JD requirements and their assessment.",
                    items=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "Requirement": types.Schema(type=types.Type.STRING, description="The specific requirement from the JD."),
                            "Score_5_0": types.Schema(type=types.Type.NUMBER, description="Score from 0.0 to 5.0."),
                            "Justification": types.Schema(type=types.Type.STRING, description="Concise justification for the score based on the resume evidence.")
                        },
                        required=["Requirement", "Score_5_0", "Justification"]
                    )
                ),
                "Hiring_Recommendation": types.Schema(type=types.Type.STRING, description="Overall summary of candidate suitability (e.g., 'Strong Hire', 'Interview Recommended', 'Do Not Proceed')."),
                "Top_3_Gaps": types.Schema(
                    type=types.Type.ARRAY,
                    description="Top 3 to 5 critical areas where the candidate's resume is deficient relative to the JD.",
                    items=types.Schema(type=types.Type.STRING)
                )
            },
            required=["Key_Requirements_Evaluation", "Hiring_Recommendation", "Top_3_Gaps"]
        )

        # 2. Simplified prompt to focus the model on the task and JSON adherence
        prompt=f"""
        Analyze the Candidate Resume against the Job Description. Your entire response MUST be a valid JSON object matching the provided schema.

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
        return response.text
        
    except GeminiAPIError as e:
        st.error(f"Gemini API Error: Could not generate report. Please check your API key and permissions. Details: {e}")
        # Return structured error message
        return json.dumps({"error": "API Error: Report generation failed.", "details": str(e)})
    except Exception as e:
        st.error(f"An unexpected error occurred during report generation: {e}")
        return json.dumps({"error": "Unexpected Error: Report generation failed.", "details": str(e)})


def extract_scores(text):
    """
    Extracts scores from the JSON report.
    Returns scores as a list of floats. Returns empty list if parsing fails.
    """
    try:
        report_data = json.loads(text)
        scores = []
        for item in report_data.get("Key_Requirements_Evaluation", []):
            score = item.get("Score_5_0")
            if score is not None:
                scores.append(float(score))
        return scores
    except json.JSONDecodeError:
        # If the input text isn't valid JSON, return empty list
        return []
    except Exception:
        # Handle other potential errors
        return []

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
    
    score_place.info("Step 2/2: Generating Structured Analysis Report (This may take up to 30 seconds)...")
    
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
        st.caption("Average of individual requirement scores (out of 5) extracted from the structured report.")
    
    st.markdown("---")

    # --- Display Detailed Report in JSON Format ---
    
    st.subheader("AI Analyst's Structured Evaluation Report (JSON)")
    
    try:
        # Attempt to load the report as JSON for pretty printing
        parsed_json = json.loads(report)
        st.json(parsed_json)
        
        # Display key insights clearly above the JSON block
        st.markdown(f"**Overall Recommendation:** **{parsed_json.get('Hiring_Recommendation', 'N/A')}**")
        st.markdown(f"**Top Gaps:** {', '.join(parsed_json.get('Top_3_Gaps', ['N/A']))}")
    except json.JSONDecodeError:
        # If the report is an error message or invalid JSON, display as text
        st.error("Error: Could not parse report as JSON. Displaying raw text/error message:")
        st.code(report)

    st.markdown("---")

    # --- Action Buttons ---
    
    st.subheader("Report Actions")
    
    col_d, col_r, _ = st.columns([1, 1, 4])
    
    with col_d:
        # Download Button
        st.download_button(
            label="Download JSON Report",
            data=report,
            file_name="Candidate_Evaluation_Report.json",
            mime="application/json",
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
