# 🔍 Intelligent Resume Parser & Shortlisting System

This project is a **smart resume parsing and shortlisting tool** built in Python that:

- Extracts structured data from resumes (PDFs)
- Parses and analyzes name, email, phone, skills, experience, education, location, and more
- Uses **spaCy**, **sentence-transformers**, **OpenAI GPT**, and **fuzzy matching**
- Matches resumes against job descriptions based on role, skills, and experience
- Saves all results in a **PostgreSQL** database

---

## 🚀 Features

- ✅ Accurate resume parsing with NLP (spaCy)
- ✅ Semantic and rule-based matching using Sentence Transformers and fuzzy logic
- ✅ Optional OpenAI GPT integration for advanced scoring (structure, certifications)
- ✅ Resume anonymization for safe processing
- ✅ Outputs all and shortlisted resumes separately
- ✅ Logs detailed output and scores
- ✅ Stores parsed data in PostgreSQL

---

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/krina-cg/CV-Parser-Scoring-.git
   cd app.py
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

 

---

## 🛢️ PostgreSQL Setup

Make sure PostgreSQL is installed and running.

Create a new database:
```sql
CREATE DATABASE resume_db;
```

Update your `DATABASE_URL` in `db.py` if needed:
```python
DATABASE_URL = "postgresql://postgres:yourpassword@localhost:5433/resume_db"
```

Then run:
```bash
python db.py
```

This will initialize the `resumes` table.

---

## 🔑 OpenAI API Key

If you're using OpenAI features (for semantic role fit scoring), set your API key:

### On Mac/Linux
```bash
export OPENAI_API_KEY="sk-..."
```

### On Windows
```bash
set OPENAI_API_KEY="sk-..."
```

---

## 📦 Prepare Resumes

Place all `.pdf` resumes into a `resumes.zip` file and put it in the project root directory.

---

## ▶️ Run the Parser

```bash
python app.py
```

Follow the prompts to:
- Enter job description
- Define job role
- Provide required and optional skills
- Set experience range 

---

## 📂 Output

The tool will create:

- `output/all_profiles_TIMESTAMP.json` — all parsed resumes
- `output/matching_profiles_TIMESTAMP.json` — shortlisted resumes
- `output/shortlisted_resumes/` — copies of shortlisted PDFs
- `output/to_be_approved_by_hr/` — borderline cases
- `logs/` — full log files with scoring breakdown

---

## 🧠 Tech Stack

- Python 3.12
- spaCy (`en_core_web_trf`)
- PyPDF2
- Sentence Transformers (`all-MiniLM-L6-v2`)
- OpenAI GPT-3.5 (optional)
- PostgreSQL + SQLAlchemy
- Fuzzy Matching (RapidFuzz)

---

## 👨‍💻 Author

Developed by Shreyash
