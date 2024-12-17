import re
import PyPDF2
import streamlit as st
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

AZURE_KEY = "4xp4YWc76c0lkREJ0fmAMlV8Y6vVOTxAdoruEVGhzRylzRzu6W9VJQQJ99ALAC1i4TkXJ3w3AAAEACOGP3ow"  
AZURE_ENDPOINT = "https://skillgapanalysis.cognitiveservices.azure.com/"  

def get_azure_client():
    return TextAnalyticsClient(endpoint=AZURE_ENDPOINT, credential=AzureKeyCredential(AZURE_KEY))

def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

def extract_text_from_txt(file):
    try:
        return file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Error extracting text from TXT: {e}")
        return None

def extract_key_phrases(text):
    try:
        client = get_azure_client()
        response = client.extract_key_phrases(documents=[text])
        key_phrases = response[0].key_phrases if response else []
        return filter_technical_skills(key_phrases)
    except Exception as e:
        st.error(f"Error extracting key phrases: {e}")
        return []

def filter_technical_skills(key_phrases):
    technical_keywords = set([
        "python", "java", "c++", "javascript", "sql", "html", "css", "aws", "azure", "docker", "kubernetes",
        "machine learning", "data science", "artificial intelligence", "deep learning", "tensorflow", "pytorch",
        "react", "angular", "node.js", "django", "flask", "git", "linux", "unix", "bash", "shell scripting",
        "agile", "scrum", "project management", "devops", "cloud computing", "big data", "hadoop", "spark"
    ])
    return [phrase for phrase in key_phrases if any(keyword in phrase.lower() for keyword in technical_keywords)]

# Function to clean text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special characters
    text = text.strip()  # Remove leading and trailing whitespace
    return text

# Function to clean a list of skills
def clean_skills(skills):
    return [clean_text(skill) for skill in skills]

def compute_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(vectors[0], vectors[1])
    return similarity[0][0] * 100  # Convert to percentage

st.title("Dynamic Skill Gap Analysis Tool")
st.markdown("Upload your resume and paste a job description to analyze the skill gap.")

uploaded_file = st.file_uploader("Upload your Resume (PDF or TXT)", type=["pdf", "txt"])

job_description = st.text_area("Paste Job Description Here")

if st.button("Analyze"):
    if uploaded_file and job_description.strip():
        
        if uploaded_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "text/plain":
            resume_text = extract_text_from_txt(uploaded_file)
        
        if resume_text:
            # Extract and clean skills from resume
            resume_skills = extract_key_phrases(resume_text)
            clean_resume_skills = clean_skills(resume_skills)
            
            # Extract and clean skills from job description
            job_description_skills = extract_key_phrases(job_description)
            clean_job_description_skills = clean_skills(job_description_skills)
            
            # Compute similarity
            similarity_percentage = compute_similarity(' '.join(clean_resume_skills), ' '.join(clean_job_description_skills))
            
            # Display results
            st.write("Resume Skills:", ', '.join(clean_resume_skills))
            st.write("Job Description Skills:", ', '.join(clean_job_description_skills))
            st.write(f"Skill Match Similarity: {similarity_percentage:.2f}%")
