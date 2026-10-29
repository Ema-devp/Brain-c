import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from PIL import Image

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Brain Scan AI Analyzer",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Brain Scan AI Analysis Assistant")
st.write("Upload a brain X-ray/MRI image and provide clinical context to generate a concise summary report.")

# Medical Disclaimer Banner
st.warning(
    "⚠️ **Medical Disclaimer**: This AI tool is intended strictly for educational and informational purposes. "
    "It does NOT offer medical diagnoses or replace evaluation by a qualified radiologist or physician."
)

# Sidebar - API Key & Settings
with st.sidebar:
    st.header("Settings")
    
    # Get API key from .env or manual user input
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        api_key = st.text_input("Enter Gemini API Key:", type="password")
    
    # Corrected Model names compatible with google-genai SDK
    selected_model = st.selectbox(
        "AI Model",
        ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"]
    )

# Layout: Upload & Form
col1, col2 = st.columns([1, 1])

# File Size Limit Configuration (15 MB)
MAX_FILE_SIZE_MB = 15
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

with col1:
    st.subheader("1. Upload Brain Scan Image")
    uploaded_file = st.file_uploader(
        f"Choose an image (JPG, JPEG, PNG, JFIF, WEBP, BMP) — Max {MAX_FILE_SIZE_MB}MB", 
        type=["jpg", "jpeg", "png", "jfif", "webp", "bmp"]
    )
    
    image = None
    if uploaded_file is not None:
        file_size = uploaded_file.size
        if file_size > MAX_FILE_SIZE_BYTES:
            st.error(f"❌ File size exceeds the {MAX_FILE_SIZE_MB} MB limit ({file_size / (1024*1024):.2f} MB). Please upload a smaller image.")
        else:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Brain Scan", use_container_width=True)
            except Exception as e:
                st.error("Failed to load image. Please upload a valid image file.")

with col2:
    st.subheader("2. Patient & Scan Context")
    scan_type = st.selectbox(
        "Scan Type", 
        ["MRI (T1/T2/FLAIR)", "CT Scan", "X-Ray", "Unknown/Other"]
    )
    symptoms = st.text_area(
        "Patient Symptoms / Clinical History", 
        placeholder="e.g., Severe headaches for 2 weeks, dizziness, double vision..."
    )
    patient_age = st.number_input("Patient Age", min_value=0, max_value=120, value=35)
    patient_gender = st.selectbox("Patient Gender", ["Not Specified", "Male", "Female", "Other"])
    
    analyze_button = st.button("🔍 Generate AI Analysis", type="primary", use_container_width=True)

# Analysis Processing
if analyze_button:
    if not api_key:
        st.error("Please provide a valid Gemini API Key in the `.env` file or sidebar.")
    elif uploaded_file is None or image is None:
        st.error("Please upload a valid brain scan image under 15MB first.")
    else:
        with st.spinner("Analyzing scan and generating concise summary..."):
            try:
                # Initialize Gemini client using official Google GenAI SDK
                client = genai.Client(api_key=api_key)

                # Prompt restricted strictly to ~10-15 lines
                prompt = f"""
                You are an expert AI radiologist assistant. Analyze the provided brain scan along with the clinical context.

                --- PATIENT CONTEXT ---
                - Scan Type: {scan_type}
                - Age: {patient_age}
                - Gender: {patient_gender}
                - Symptoms: {symptoms if symptoms else "None provided"}

                --- CRITICAL RESPONSE INSTRUCTIONS ---
                Keep your complete output concise (strictly around 10 to 15 lines total).
                Structure your output into short bullet points covering:
                1. Visual anatomical findings & scan plane/view.
                2. Potential clinical considerations or visual irregularities observed.
                3. Key diagnostic recommendations/next steps for the medical team.
                """

                # Call API
                response = client.models.generate_content(
                    model=selected_model,
                    contents=[image, prompt]
                )

                st.subheader("📊 Analysis Results")
                st.markdown(response.text)

            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")