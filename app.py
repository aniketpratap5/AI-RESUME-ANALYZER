from flask import Flask, render_template, request
import pdfplumber
import docx2txt
import os
import json
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from skill_keywords import SKILLS

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load jobs
with open('jobs.json', 'r') as file:
    JOBS = json.load(file)

# Extract text from PDF

def extract_pdf_text(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + " "
    return text.lower()

# Extract text from DOCX

def extract_docx_text(path):
    return docx2txt.process(path).lower()

# Extract skills

def extract_skills(text):
    found_skills = []

    for skill in SKILLS:
        if skill.lower() in text:
            found_skills.append(skill)

    return list(set(found_skills))

# Calculate ATS score

def calculate_ats_score(skills):
    total_skills = len(SKILLS)
    matched_skills = len(skills)

    score = int((matched_skills / total_skills) * 100)

    return min(score, 100)

# Match jobs

def match_jobs(resume_skills):
    matched_jobs = []

    for job in JOBS:
        job_skills = job['skills']

        documents = [
            ' '.join(resume_skills),
            ' '.join(job_skills)
        ]

        cv = CountVectorizer()
        matrix = cv.fit_transform(documents)

        similarity = cosine_similarity(matrix)[0][1]

        missing_skills = list(set(job_skills) - set(resume_skills))

        matched_jobs.append({
            'title': job['title'],
            'company': job['company'],
            'location': job['location'],
            'job_type': job['type'],
            'apply_link': job['apply_link'],
            'match_percentage': round(similarity * 100, 2),
            'required_skills': job_skills,
            'missing_skills': missing_skills
        })

    matched_jobs = sorted(
        matched_jobs,
        key=lambda x: x['match_percentage'],
        reverse=True
    )

    return matched_jobs

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['resume']

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Extract text
            if filename.endswith('.pdf'):
                text = extract_pdf_text(filepath)
            elif filename.endswith('.docx'):
                text = extract_docx_text(filepath)
            else:
                return "Unsupported file format"

            # Analyze
            skills = extract_skills(text)
            ats_score = calculate_ats_score(skills)
            matched_jobs = match_jobs(skills)

            return render_template(
                'index.html',
                skills=skills,
                ats_score=ats_score,
                jobs=matched_jobs
            )

    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
