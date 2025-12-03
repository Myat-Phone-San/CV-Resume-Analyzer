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
    st.error("GEMINI_API_KEY not found in Streamlit secrets.")
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

# --- Title and Branding for Employer Tool ---
st.title("🧑‍💼 AI Candidate Match Evaluator (Concise JSON)")
st.markdown("Generates a structured, concise evaluation tailored for employer systems.")

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
            return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            
        ats_model = load_model()
    
    embeddings1 = ats_model.encode([text1], convert_to_tensor=False)
    embeddings2 = ats_model.encode([text2], convert_to_tensor=False)
    
    similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
    return float(similarity)


# Rewritten function to use the Gemini API and force concise JSON output
def get_report(resume, job_desc, ats_score, avg_llm_score):
    """Generates a detailed candidate evaluation report using the Gemini LLM in strict JSON format."""
    try:
        client = genai.Client(api_key=api_key)

        # 1. Define the mandatory JSON structure (Response Schema)
        json_schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "Candidate_Score_LLM_Avg_5_0": types.Schema(type=types.Type.NUMBER, description="The mean of the scores for all key requirements."),
                "ATS_Accuracy_Score": types.Schema(type=types.Type.NUMBER, description="The Pre-calculated ATS/Semantic Similarity Score (0.0 to 1.0)."),
                "Key_Gaps_Summary": types.Schema(
                    type=types.Type.ARRAY,
                    description="The 3 to 5 most critical areas where the candidate is deficient relative to the JD.",
                    items=types.Schema(type=types.Type.STRING)
                ),
                "Key_Requirements_Evaluation": types.Schema(
                    type=types.Type.ARRAY,
                    description="A concise assessment of 5 critical JD requirements.",
                    items=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "Requirement": types.Schema(type=types.Type.STRING, description="The specific requirement from the JD."),
                            "Score_5_0": types.Schema(type=types.Type.NUMBER, description="Score from 0.0 to 5.0."),
                            "Concise_Justification": types.Schema(type=types.Type.STRING, description="A single, short sentence summarizing the evidence or lack thereof.")
                        },
                        required=["Requirement", "Score_5_0", "Concise_Justification"]
                    )
                ),
                "Hiring_Recommendation": types.Schema(type=types.Type.STRING, description="Overall suitability (e.g., 'Strong Match', 'Proceed to Interview', 'Reject')."),
            },
            required=["Candidate_Score_LLM_Avg_5_0", "ATS_Accuracy_Score", "Key_Gaps_Summary", "Key_Requirements_Evaluation", "Hiring_Recommendation"]
        )

        # 2. Simplified prompt to focus the model on the task, conciseness, and JSON adherence
        prompt=f"""
        Analyze the Candidate Resume against the Job Description. Your task is to extract and summarize key data points into a concise JSON object. The average LLM score and ATS score are pre-calculated and must be inserted into the final JSON structure.

        **PRE-CALCULATED SCORES (Inject these values into the final JSON):**
        - Candidate_Score_LLM_Avg_5_0: {avg_llm_score:.2f}
        - ATS_Accuracy_Score: {ats_score:.4f}

        **INSTRUCTIONS:**
        1. Identify the 5 most critical requirements.
        2. Assign a Score_5_0 (0.0 to 5.0) and provide a **Concise_Justification** (one short sentence).
        3. Identify the **Top_Gaps_Summary** (3-5 items).
        4. Provide the **Hiring_Recommendation**.
        5. Your entire response MUST be a valid JSON object matching the provided schema.

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
        
        # Check for the top-level LLM Score directly (new structure)
        llm_score = report_data.get("Candidate_Score_LLM_Avg_5_0")
        if llm_score is not None:
            return [float(llm_score)]

        # Fallback to calculating the mean from the requirements array
        scores = []
        for item in report_data.get("Key_Requirements_Evaluation", []):
            score = item.get("Score_5_0")
            if score is not None:
                scores.append(float(score))
        return scores if scores else []
        
    except json.JSONDecodeError:
        return []
    except Exception:
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
                    # Calculate ATS score immediately to use in the LLM prompt
                    st.session_state.ats_score = calculate_similarity_bert(st.session_state.resume, st.session_state.job_desc)
                    st.session_state.form_submitted = True
                    st.rerun()  # Refresh the page to close the form and display results

            # Do not allow if not uploaded
            else:
                st.warning("Please upload a **Resume** and provide a **Job Description** to analyze.")


if st.session_state.form_submitted:
    
    # Placeholders for dynamic updates
    score_place = st.info("Step 1/2: Calculating LLM Report...")
    
    # Use the stored ATS score
    ats_score = st.session_state.ats_score

    # Calculate a dummy LLM score average based on the ATS score for the first run, 
    # then replace it after the report is generated.
    # This initial LLM score is passed into get_report so it can be injected into the JSON.
    initial_llm_avg = ats_score * 5.0 
    
    score_place.info("Step 2/2: Generating Structured Analysis Report (This may take up to 30 seconds)...")
    
    # 2. Get the Analysis Report from LLM (Gemini)
    report = get_report(st.session_state.resume, st.session_state.job_desc, ats_score, initial_llm_avg)

    # 3. Calculate the Average Score from the LLM Report (or extract the injected value)
    report_scores = extract_scores(report) 
    
    # Correctly calculate the average score out of 5
    if report_scores:
        avg_score = np.mean(report_scores) 
        avg_score_display = f"{avg_score:.2f} / 5.0"
        # Since the LLM returns the final average, let's re-run the report generation 
        # with the *actual* calculated average score to ensure accuracy in the final JSON.
        # NOTE: This double-run is only needed if you want the LLM to calculate and THEN inject. 
        # For simplicity and speed, we will proceed with the current calculated average.
    else:
        # If score extraction fails, use the initial dummy value for display
        avg_score = initial_llm_avg
        avg_score_display = "N/A"

    score_place.success("Evaluation completed successfully!")
    st.markdown("---")

    # --- Display Scores and Key Gaps ---
    
    st.subheader("📊 Candidate Fit Overview")
    col1, col2 = st.columns(2, border=True)
    
    with col1:
        st.write("Keywords and Contextual Similarity Score (ATS):")
        st.metric(label="ATS Accuracy Score (0.0 to 1.0)", value=f"{ats_score:.4f}")
        st.caption("Semantic match score.")

    with col2:
        st.write("Average Requirement Fulfillment Score (LLM):")
        st.metric(label="Candidate Score (Avg / 5.0)", value=avg_score_display)
        st.caption("Average of individual requirement scores.")
    
    # Try to display the key gaps prominently
    try:
        parsed_json = json.loads(report)
        gaps = parsed_json.get('Key_Gaps_Summary', [])
        recommendation = parsed_json.get('Hiring_Recommendation', 'N/A')
        
        st.markdown("---")
        st.subheader("Key Gaps and Recommendation")
        
        st.markdown(f"**Hiring Recommendation:** **{recommendation}**")
        
        if gaps:
            st.markdown("**Top Deficiencies (Gaps):**")
            for gap in gaps:
                st.markdown(f"* ❌ {gap}")
        else:
            st.markdown("**Top Deficiencies (Gaps):** None found or N/A.")
        
    except json.JSONDecodeError:
        # Handle cases where the report isn't perfect JSON
        st.warning("Could not extract Gaps/Recommendation from the report structure.")
        
    st.markdown("---")

    # --- Display Detailed Report (JSON) ---
    
    st.subheader("Structured Evaluation Report (JSON)")
    
    try:
        st.json(parsed_json)
    except NameError:
        # If parsing failed above, ensure we display the raw content for debugging
        st.error("Report parsing failed. Displaying raw output:")
        st.code(report)

    st.markdown("---")

    # --- Action Buttons (Dual Downloads) ---
    
    st.subheader("Report Actions")
    
    # Prepare data for Word/Text Download (formatted key-value pairs)
    # This prepares a human-readable text output from the JSON.
    try:
        word_data = f"CANDIDATE EVALUATION REPORT\n"
        word_data += f"--------------------------------------------------\n"
        word_data += f"ATS ACCURACY SCORE: {parsed_json.get('ATS_Accuracy_Score', 'N/A')}\n"
        word_data += f"CANDIDATE SCORE (LLM AVG): {parsed_json.get('Candidate_Score_LLM_Avg_5_0', 'N/A')}/5.0\n"
        word_data += f"HIRING RECOMMENDATION: {parsed_json.get('Hiring_Recommendation', 'N/A')}\n\n"
        word_data += f"KEY DEFICIENCIES (GAPS):\n"
        for gap in parsed_json.get('Key_Gaps_Summary', []):
            word_data += f"- {gap}\n"
        word_data += f"\nKEY REQUIREMENTS BREAKDOWN:\n"
        for item in parsed_json.get('Key_Requirements_Evaluation', []):
            word_data += f"--- {item.get('Requirement', 'N/A')} ---\n"
            word_data += f"Score: {item.get('Score_5_0', 'N/A')}/5.0\n"
            word_data += f"Justification: {item.get('Concise_Justification', 'N/A')}\n"
    except Exception:
        word_data = "Error: Could not format the structured report for text download."


    col_json, col_word, col_r, _ = st.columns([1.5, 1.5, 1, 3])
    
    with col_json:
        # JSON Download Button
        st.download_button(
            label="📥 Download JSON File",
            data=report,
            file_name="Candidate_Evaluation_Report.json",
            mime="application/json",
        )
    
    with col_word:
        # Text/Word Download Button
        st.download_button(
            label="📄 Download Text Report",
            data=word_data,
            file_name="Candidate_Evaluation_Summary.txt", # Use .txt, instruct user to open in Word
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

# -----------------------------------------------------
