import os
import re
import json
import zipfile
import logging
from datetime import datetime
from PyPDF2 import PdfReader
import spacy
from spacy.matcher import PhraseMatcher
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util
import shutil
import importlib.metadata
import math
from db import save_resume_to_db

# Load spaCy model globally
nlp = spacy.load("en_core_web_trf")
# Load sentence-transformers model globally
st_model = SentenceTransformer('all-MiniLM-L6-v2')

# Configure logging
def setup_logging():
    """Setup logging configuration"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'resume_parser_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # This will also print to console
        ]
    )
    return log_file

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {str(e)}")
        return ""

def extract_email(text):
    """Extract email address from text."""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    match = re.search(email_pattern, text)
    return match.group(0) if match else ""

def extract_phone(text):
    """Extract phone number from text."""
    phone_patterns = [
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        r'\+\d{1,3}[-.]?\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        r'\b\d{10}\b'
    ]
    
    for pattern in phone_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return ""

def extract_name(text):
    doc = nlp(text)
    skills_and_techs = set([
        'python', 'java', 'javascript', 'oracle', 'aws', 'react', 'django', 'mysql', 'postgresql', 'css', 'html', 'nodejs', 'typescript', 'r', 'ai', 'data science',
        'c++', 'c#', 'php', 'swift', 'kotlin', 'go', 'golang', 'rust', 'scala', 'perl', 'matlab', 'sass', 'scss', 'angular', 'vue', 'node.js', 'express',
        'spring', 'asp.net', 'laravel', 'symfony', 'jquery', 'bootstrap', 'tailwind', 'next.js', 'nuxt.js', 'gatsby', 'redis', 'cassandra', 'elasticsearch',
        'oracle', 'sqlite', 'dynamodb', 'firebase', 'neo4j', 'couchdb', 'amazon web services', 'azure', 'gcp', 'google cloud', 'heroku', 'digitalocean',
        'cloudflare', 'alibaba cloud', 'ibm cloud', 'docker', 'kubernetes', 'k8s', 'jenkins', 'git', 'github', 'gitlab', 'bitbucket', 'terraform', 'ansible',
        'puppet', 'chef', 'prometheus', 'grafana', 'elk stack', 'ci/cd', 'cicd', 'continuous integration', 'continuous deployment', 'machine learning',
        'ml', 'artificial intelligence', 'ai', 'deep learning', 'dl', 'neural networks', 'nn', 'tensorflow', 'pytorch', 'keras', 'scikit-learn',
        'computer vision', 'cv', 'natural language processing', 'nlp', 'agile', 'scrum', 'kanban', 'waterfall', 'devops', 'lean', 'six sigma',
        'test driven development', 'tdd', 'behavior driven development', 'bdd'
    ])
    # 1. Try spaCy NER
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            candidate = ent.text.strip()
            if candidate.lower() not in skills_and_techs and len(candidate.split()) >= 1:
                return candidate
    # 2. Try regex for "Name:"
    name_match = re.search(r'(?i)(?:name:?[ \t]*)([A-Za-z\s]+)', text)
    if name_match:
        candidate = name_match.group(1).strip()
        if candidate.lower() not in skills_and_techs and len(candidate.split()) >= 1:
            return candidate
    # 3. Try first non-empty line
    for line in text.split('\n'):
        line = line.strip()
        if line and line.lower() not in skills_and_techs and all(c.isalpha() or c.isspace() for c in line):
            return line
    # 4. Fallback to email username
    email = extract_email(text)
    if email:
        username = email.split('@')[0]
        if '.' in username:
            return ' '.join([part.capitalize() for part in username.split('.')])
        return username.capitalize()
    return "Unknown"

def normalize_text_spacy(text):
    """Lemmatize and tokenize text using spaCy, removing stopwords and punctuation."""
    doc = nlp(text)
    return set(token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct)

def extract_skills(text):
    """Extract skills from text using spaCy for normalization and comprehensive keywords."""
    skills_categories = {
        'programming_languages': [
            'python', 'java', 'javascript', 'js', 'typescript', 'ts', 'c++', 'cpp', 'c#', 'csharp',
            'ruby', 'php', 'swift', 'kotlin', 'go', 'golang', 'rust', 'scala', 'perl', 'r', 'matlab'
        ],
        'web_technologies': [
            'html', 'css', 'sass', 'scss', 'react', 'angular', 'vue', 'node.js', 'nodejs', 'express',
            'django', 'flask', 'spring', 'asp.net', 'laravel', 'symfony', 'jquery', 'bootstrap',
            'tailwind', 'next.js', 'nuxt.js', 'gatsby'
        ],
        'databases': [
            'sql', 'mysql', 'postgresql', 'postgres', 'mongodb', 'redis', 'cassandra', 'elasticsearch',
            'oracle', 'sqlite', 'dynamodb', 'firebase', 'neo4j', 'couchdb'
        ],
        'cloud_platforms': [
            'aws', 'amazon web services', 'azure', 'gcp', 'google cloud', 'heroku', 'digitalocean',
            'cloudflare', 'alibaba cloud', 'ibm cloud'
        ],
        'devops_tools': [
            'docker', 'kubernetes', 'k8s', 'jenkins', 'git', 'github', 'gitlab', 'bitbucket',
            'terraform', 'ansible', 'puppet', 'chef', 'prometheus', 'grafana', 'elk stack',
            'ci/cd', 'cicd', 'continuous integration', 'continuous deployment'
        ],
        'ai_ml': [
            'machine learning', 'ml', 'artificial intelligence', 'ai', 'deep learning', 'dl',
            'neural networks', 'nn', 'tensorflow', 'pytorch', 'keras', 'scikit-learn',
            'computer vision', 'cv', 'natural language processing', 'nlp', 'data science'
        ],
        'methodologies': [
            'agile', 'scrum', 'kanban', 'waterfall', 'devops', 'lean', 'six sigma',
            'test driven development', 'tdd', 'behavior driven development', 'bdd'
        ]
    }
    found_skills = []
    tokens = normalize_text_spacy(text)
    for category, skills in skills_categories.items():
        for skill in skills:
            # For multi-word skills, check if all words are present
            if ' ' in skill:
                if all(word in tokens for word in skill.split()):
                    found_skills.append(skill)
            else:
                if skill in tokens:
                    found_skills.append(skill)
    return list(set(found_skills))

def extract_education(text):
    """Extract structured education information: degree, institution, year."""
    # Look for degree + institution + year patterns
    degree_pattern = r'(?i)(Bachelor|Master|PhD|B\.S\.|M\.S\.|B\.E\.|M\.E\.|B\.Tech|M\.Tech|MBA|BBA|BCA|MCA)[^\n,;]*'
    institution_pattern = r'(?i)(University|College|Institute|School|Academy)[^\n,;]*'
    year_pattern = r'(19|20)\d{2}'
    education = []
    lines = text.split('\n')
    for line in lines:
        deg = re.search(degree_pattern, line)
        inst = re.search(institution_pattern, line)
        year = re.search(year_pattern, line)
        if deg or inst:
            entry = []
            if deg:
                entry.append(deg.group(0).strip())
            if inst:
                entry.append(inst.group(0).strip())
            if year:
                entry.append(year.group(0))
            if entry:
                education.append(", ".join(entry))
    return list(set(education))

def split_sections(text):
    """Split resume text into sections based on common headers."""
    section_headers = [
        'experience', 'work experience', 'professional experience', 'skills', 'technical skills', 'education', 'academic background',
        'projects', 'certifications', 'summary', 'objective', 'profile', 'personal information', 'contact', 'interests', 'hobbies'
    ]
    sections = {}
    current_section = 'other'
    lines = text.split('\n')
    for line in lines:
        header = line.strip().lower().rstrip(':')
        if header in section_headers:
            current_section = header
            sections[current_section] = []
        else:
            if current_section not in sections:
                sections[current_section] = []
            sections[current_section].append(line)
    # Join lines for each section
    for k in sections:
        sections[k] = '\n'.join(sections[k]).strip()
    return sections

def extract_keywords(text, extra_phrases=None):
    """Section-aware: prioritize keywords from Experience, Skills, Education sections."""
    sections = split_sections(text)
    prioritized_sections = ['experience', 'work experience', 'professional experience', 'skills', 'technical skills', 'education', 'academic background']
    prioritized_text = ''
    for sec in prioritized_sections:
        if sec in sections:
            prioritized_text += sections[sec] + '\n'
    # If no prioritized sections found, use whole text
    if not prioritized_text.strip():
        prioritized_text = text
    # Now use the improved filtering logic on prioritized_text
    doc = nlp(prioritized_text)
    keywords = set()
    generic_words = set([
        'experience', 'work', 'project', 'projects', 'team', 'teams', 'skills', 'skill', 'education', 'summary', 'objective',
        'software', 'development', 'developer', 'engineer', 'company', 'companies', 'role', 'responsibilities', 'responsibility',
        'year', 'years', 'month', 'months', 'job', 'jobs', 'resume', 'curriculum', 'vitae', 'cv', 'profile', 'contact', 'address',
        'email', 'phone', 'linkedin', 'github', 'technologies', 'tools', 'languages', 'language', 'certification', 'certifications',
        'degree', 'degrees', 'bachelor', 'master', 'phd', 'msc', 'bsc', 'mba', 'school', 'college', 'university', 'institute',
        'location', 'city', 'state', 'country', 'zip', 'postal', 'code', 'date', 'dates', 'period', 'duration', 'intern', 'internship',
        'full-time', 'part-time', 'contract', 'permanent', 'temporary', 'current', 'previous', 'former', 'present', 'future', 'past',
        'objective', 'summary', 'description', 'details', 'detail', 'reference', 'references', 'hobbies', 'interests', 'personal', 'information'
    ])
    for ent in doc.ents:
        if ent.label_ in {"ORG", "PRODUCT", "GPE", "WORK_OF_ART", "EVENT", "LAW", "LANGUAGE", "FAC", "NORP", "PERSON"}:
            keywords.add(ent.text.strip().lower())
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.strip().lower()
        if len(chunk_text) > 2 and chunk_text not in generic_words and not chunk_text.isdigit():
            keywords.add(chunk_text)
    for token in doc:
        t = token.lemma_.lower()
        if (
            token.pos_ in {"NOUN", "PROPN", "ADJ"}
            and not token.is_stop
            and not token.is_punct
            and len(t) > 2
            and t not in generic_words
            and not t.isdigit()
            and t.isalpha()
        ):
            keywords.add(t)
    if extra_phrases:
        matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        patterns = [nlp.make_doc(phrase) for phrase in extra_phrases]
        matcher.add("EXTRA_PHRASES", patterns)
        matches = matcher(doc)
        for match_id, start, end in matches:
            phrase = doc[start:end].text.strip().lower()
            if len(phrase) > 2 and phrase not in generic_words and not phrase.isdigit():
                keywords.add(phrase)
    return list(keywords)

def extract_experience(text):
    """Section-aware: prioritize experience from Experience/Work sections."""
    sections = split_sections(text)
    prioritized_sections = ['experience', 'work experience', 'professional experience']
    prioritized_text = ''
    for sec in prioritized_sections:
        if sec in sections:
            prioritized_text += sections[sec] + '\n'
    if not prioritized_text.strip():
        prioritized_text = text
    experience_patterns = [
        r'(?i)(Senior|Junior|Lead|Principal|Staff)?\s*(Software|Web|Mobile|Full Stack|Backend|Frontend)\s*(Engineer|Developer|Architect)',
        r'(?i)(Senior|Junior|Lead|Principal|Staff)?\s*(DevOps|Cloud|Security|QA|Test|Automation)\s*(Engineer|Developer|Architect)',
        r'(?i)(Senior|Junior|Lead|Principal|Staff)?\s*(Data|Machine Learning|AI|ML|Deep Learning)\s*(Engineer|Scientist|Analyst|Architect)',
        r'(?i)(Senior|Junior|Lead|Principal|Staff)?\s*(Business Intelligence|BI|Data Warehouse|ETL)\s*(Engineer|Developer|Analyst)',
        r'(?i)(Senior|Junior|Lead|Principal|Staff)?\s*(Project|Technical|Team|Product|Program|Engineering)\s*(Manager|Lead|Architect|Director)',
        r'(?i)(Senior|Junior|Lead|Principal|Staff)?\s*(Scrum|Agile|DevOps|Release|Build)\s*(Master|Manager|Lead)',
        r'(?i)(Senior|Junior|Lead|Principal|Staff)?\s*(UI|UX|User Interface|User Experience)\s*(Designer|Developer|Engineer|Architect)',
        r'(?i)(Senior|Junior|Lead|Principal|Staff)?\s*(Database|DB|System|Network|Security|Cloud)\s*(Administrator|Admin|Engineer|Architect)',
        r'(?i)(Senior|Junior|Lead|Principal|Staff)?\s*(Game|Mobile|Embedded|Firmware|Hardware)\s*(Developer|Engineer|Architect)',
        r'(?i)(Senior|Junior|Lead|Principal|Staff)?\s*(Blockchain|Cryptocurrency|Smart Contract)\s*(Developer|Engineer|Architect)',
        # Add more patterns for other domains as needed
        r'(?i)(Accountant|CPA|Bookkeeper|Financial Analyst|Controller|Auditor|Tax Specialist|Payroll Specialist|Finance Manager|Treasurer|Cost Accountant|Accounts Payable|Accounts Receivable)'
    ]
    experience = []
    for pattern in experience_patterns:
        matches = re.finditer(pattern, prioritized_text)
        for match in matches:
            role = match.group(0).strip()
            role = re.sub(r'\s+', ' ', role)
            role = role.replace('  ', ' ')
            experience.append(role)
    return list(set(experience))

def extract_years_experience(text):
    """Extract total years of experience from resume text with comprehensive pattern matching."""
    
    years = []
    text_lower = text.lower()
    
    # 1. Direct experience patterns
    experience_patterns = [
        # Standard formats
        r'(\d+(?:\.\d+)?)\s*\+?\s*years?\s+(?:of\s+)?experience',
        r'experience[:\s-]+(\d+(?:\.\d+)?)\s*\+?\s*years?',
        r'(\d+(?:\.\d+)?)\s*\+?\s*years?\s+(?:in|with|as|working)',
        r'(\d+(?:\.\d+)?)\s*\+?\s*years?\s+(?:professional|work|working)',
        r'over\s+(\d+(?:\.\d+)?)\s*\+?\s*years?',
        r'more\s+than\s+(\d+(?:\.\d+)?)\s*\+?\s*years?',
        r'(\d+(?:\.\d+)?)\s*\+?\s*years?\s+(?:total|overall)',
        
        # Range formats
        r'(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*\+?\s*years?',
        r'between\s+(\d+(?:\.\d+)?)\s*(?:and|to|-|–)\s*(\d+(?:\.\d+)?)\s*years?',
        
        # Alternative spellings
        r'(\d+(?:\.\d+)?)\s*\+?\s*yrs?\s+(?:of\s+)?(?:experience|exp)',
        r'(\d+(?:\.\d+)?)\s*\+?\s*yr\s+(?:experience|exp)',
        
        # Fractional years
        r'(\d+)\s*and\s*(?:a\s+)?half\s+years?',  # "2 and half years"
        r'(\d+)\.5\s+years?',  # "2.5 years"
        
        # Written numbers
        r'(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\s+years?\s+(?:of\s+)?experience',
    ]
    
    # Word to number mapping
    word_to_num = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
        'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20
    }
    
    # Extract years from direct patterns
    for pattern in experience_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            if isinstance(match, tuple):
                # Handle ranges - take the average or max
                valid_nums = []
                for m in match:
                    if str(m).replace('.', '').isdigit():
                        valid_nums.append(float(m))
                    elif m in word_to_num:
                        valid_nums.append(word_to_num[m])
                
                if valid_nums:
                    # For ranges, take the maximum
                    years.append(max(valid_nums))
            else:
                if str(match).replace('.', '').isdigit():
                    years.append(float(match))
                elif match in word_to_num:
                    years.append(word_to_num[match])
                elif 'half' in str(match):
                    # Handle "2 and half years"
                    num_match = re.search(r'(\d+)', str(match))
                    if num_match:
                        years.append(float(num_match.group(1)) + 0.5)
    
    # 2. Date range extraction and calculation
    current_year = datetime.now().year
    
    # Look for employment date ranges
    date_patterns = [
        # Format: Jan 2020 - Present, January 2020 - Current, etc.
        r'([a-z]{3,9})\s+(\d{4})\s*[-–]\s*(?:present|current|till\s+date|now)',
        r'(\d{1,2})[\/\-](\d{4})\s*[-–]\s*(?:present|current|till\s+date|now)',
        
        # Format: Jan 2020 - Dec 2022, 01/2020 - 12/2022, etc.
        r'([a-z]{3,9})\s+(\d{4})\s*[-–]\s*([a-z]{3,9})\s+(\d{4})',
        r'(\d{1,2})[\/\-](\d{4})\s*[-–]\s*(\d{1,2})[\/\-](\d{4})',
        
        # Format: 2020-2022, 2020 - 2022
        r'(\d{4})\s*[-–]\s*(\d{4})',
        r'(\d{4})\s*[-–]\s*(?:present|current)',
    ]
    
    month_to_num = {
        'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
        'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6,
        'jul': 7, 'july': 7, 'aug': 8, 'august': 8, 'sep': 9, 'september': 9,
        'oct': 10, 'october': 10, 'nov': 11, 'november': 11, 'dec': 12, 'december': 12
    }
    
    total_experience_months = 0
    
    for pattern in date_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            try:
                if len(match) == 2:  # Year range or month-year to present
                    if match[1].isdigit():  # Year to present
                        start_year = int(match[1])
                        experience_years = current_year - start_year
                        years.append(experience_years)
                    elif match[0].isdigit() and match[1].isdigit():  # Year range
                        start_year = int(match[0])
                        end_year = int(match[1])
                        experience_years = end_year - start_year
                        years.append(experience_years)
                
                elif len(match) == 4:  # Full date ranges
                    # Month-Year to Month-Year format
                    start_month = month_to_num.get(match[0], 1) if match[0].isalpha() else int(match[0])
                    start_year = int(match[1])
                    end_month = month_to_num.get(match[2], 12) if match[2].isalpha() else int(match[2])
                    end_year = int(match[3])
                    
                    # Calculate total months
                    months = (end_year - start_year) * 12 + (end_month - start_month)
                    if months > 0:
                        years.append(months / 12.0)
                        
            except (ValueError, KeyError):
                continue
    
    # 3. Look for experience in job descriptions
    # Find sections that might contain work experience
    experience_sections = []
    lines = text.split('\n')
    
    in_experience_section = False
    experience_keywords = ['experience', 'employment', 'work history', 'professional', 'career', 'positions held']
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Check if we're entering an experience section
        if any(keyword in line_lower for keyword in experience_keywords):
            in_experience_section = True
            continue
            
        # Check if we're leaving experience section
        if in_experience_section and (line_lower.startswith('education') or line_lower.startswith('skills') or line_lower.startswith('certifications')):
            in_experience_section = False
            
        if in_experience_section:
            experience_sections.append(line)
    
    # Look for tenure patterns in experience sections
    experience_text = ' '.join(experience_sections)
    tenure_patterns = [
        r'(\d+(?:\.\d+)?)\s*years?\s*(?:and\s*)?(\d+)?\s*months?',
        r'(\d+)\s*months?',
        r'(\d+(?:\.\d+)?)\s*years?',
    ]
    
    for pattern in tenure_patterns:
        matches = re.findall(pattern, experience_text.lower())
        for match in matches:
            if isinstance(match, tuple):
                years_part = float(match[0]) if match[0] else 0
                months_part = float(match[1]) if match[1] else 0
                total_years = years_part + (months_part / 12.0)
                years.append(total_years)
            else:
                if 'month' in pattern:
                    years.append(float(match) / 12.0)  # Convert months to years
                else:
                    years.append(float(match))
    
    # 4. Filter and return maximum
    # Remove unrealistic values (more than 50 years or less than 0)
    valid_years = [y for y in years if 0 <= y <= 50]
    
    if valid_years:
        # Return the maximum found (most likely to be total experience)
        max_years = max(valid_years)
        return int(max_years) if max_years == int(max_years) else round(max_years, 1)
    
    return 0


# Alternative helper function to debug what's being extracted
def debug_experience_extraction(text):
    """Debug function to see what experience patterns are found in the text."""
    print("=== EXPERIENCE EXTRACTION DEBUG ===")
    print(f"Text length: {len(text)} characters")
    print(f"First 500 characters: {text[:500]}...")
    
    # Test each pattern individually
    patterns_to_test = [
        (r'(\d+(?:\.\d+)?)\s*\+?\s*years?\s+(?:of\s+)?experience', "Direct experience pattern"),
        (r'(\d+{4})\s*[-–]\s*(\d{4})', "Year range pattern"),
        (r'([a-z]{3,9})\s+(\d{4})\s*[-–]\s*(?:present|current)', "Month-Year to present"),
    ]
    
    for pattern, description in patterns_to_test:
        matches = re.findall(pattern, text.lower())
        if matches:
            print(f"{description}: {matches}")
    
    result = extract_years_experience(text)
    print(f"Final extracted years: {result}")
    print("=====================================")
    return result

def extract_location(text):
    """Extract location from resume text (simple heuristic: look for city/state/country lines)."""
    # This is a naive approach; can be improved with NER or a location list
    location_pattern = r'(?i)([A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*(?:,\s*[A-Z]{2,})?)'
    lines = text.split('\n')
    for line in lines:
        if re.search(location_pattern, line):
            return line.strip()
    return ""

def parse_resume(pdf_path, job_keywords=None):
    """Parse a resume PDF and extract relevant information using spaCy for name, keywords, experience, years, and location."""
    text = extract_text_from_pdf(pdf_path)
    if not text:
        return None
    name = extract_name(text)
    keywords = extract_keywords(text, extra_phrases=job_keywords)
    experience = extract_experience(text)
    years_experience = extract_years_experience(text)
    location = extract_location(text)
    return {
        "name": name,
        "email": extract_email(text),
        "phone": extract_phone(text),
        "education": extract_education(text),
        "keywords": keywords,
        "experience": experience,
        "years_experience": years_experience,
        "location": location
    }

def is_profile_match(profile, job_role, required_skills, optional_skills, min_years, max_years, location_type, onsite_location, job_role_embedding=None):
    """Check if a profile matches the job role and requirements using all new criteria (no location in scoring, min/max years)."""
    # Role matching
    profile_experience = set(profile.get('experience', []))
    role_match_score = 0
    for exp in profile_experience:
        for role in [job_role.lower()]:
            if fuzz.ratio(exp.lower(), role) > 80:
                role_match_score = 100
                break
        if role_match_score > 0:
            break
    # Skills matching
    profile_keywords = set(profile.get('keywords', []))
    matching_skills = set()
    for skill in required_skills:
        skill_lower = skill.lower().strip()
        for pk in profile_keywords:
            if fuzz.ratio(skill_lower, pk.lower()) > 85:
                matching_skills.add(skill)
    skills_match_percentage = (len(matching_skills) / len(required_skills) * 100) if required_skills else 0
    # Optional skills
    matching_optional_skills = set()
    for skill in optional_skills:
        skill_lower = skill.lower().strip()
        for pk in profile_keywords:
            if fuzz.ratio(skill_lower, pk.lower()) > 85:
                matching_optional_skills.add(skill)
    optional_skills_percentage = (len(matching_optional_skills) / len(optional_skills) * 100) if optional_skills else 0
    # Years of experience
    years_experience = profile.get('years_experience', 0)
    too_much_exp = False
    if max_years > 0:
        if years_experience > max_years:
            too_much_exp = True
    if years_experience < min_years:
        years_score = (years_experience / min_years * 100) if min_years > 0 else 0
    elif not too_much_exp:
        years_score = 100
    else:
        years_score = 0
    years_score = math.ceil(years_score)
    # Weighted rule-based score (50% required skills, 10% optional, 15% role)
    rule_score = (
        0.5 * skills_match_percentage +
        0.10 * optional_skills_percentage +
        0.15 * role_match_score
    )
    # Semantic similarity for role context
    profile_text = ' '.join(profile.get('keywords', [])) + ' ' + ' '.join(profile.get('experience', []))
    st_semantic_score = 0.0
    openai_semantic_score = None
    if not profile_text.strip():
        semantic_score = 0.0
    else:
        profile_embedding = st_model.encode(profile_text, convert_to_tensor=True)
        if job_role_embedding is None:
            job_role_embedding = st_model.encode(job_role, convert_to_tensor=True)
        st_semantic_score = float(util.pytorch_cos_sim(profile_embedding, job_role_embedding)[0][0]) * 100
        # OpenAI semantic fit score
        anonymized_text = anonymize_resume_text(profile_text, name=profile.get('name', None))
        openai_prompt = (
            f"Given the following anonymized resume and the job role '{job_role}', rate the semantic fit of this candidate for the role on a scale of 0-100, considering certifications, structure, and overall relevance. Respond with only the number.\n\nResume:\n{anonymized_text}"
        )
        try:
            import openai
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if openai_api_key:
                openai.api_key = openai_api_key
                # Check OpenAI version
                try:
                    version = importlib.metadata.version('openai')
                except Exception:
                    version = '0.0.0'
                major_version = int(version.split('.')[0])
                openai_semantic_score = None
                try:
                    if major_version >= 1:
                        # OpenAI v1.x syntax
                        try:
                            response = openai.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[{"role": "user", "content": openai_prompt}],
                                max_tokens=10,
                                temperature=0.2
                            )
                            content = response.choices[0].message.content
                            import re
                            if isinstance(content, str):
                                match = re.search(r'\d+', content)
                                if match:
                                    openai_semantic_score = float(match.group(0))
                        except Exception:
                            openai_semantic_score = None
                    # Remove OpenAI v0.x fallback
                except Exception:
                    openai_semantic_score = None
        except Exception:
            openai_semantic_score = None
    if openai_semantic_score is not None:
        semantic_score = (st_semantic_score + openai_semantic_score) / 2
    else:
        semantic_score = st_semantic_score
    # Combine scores (60% rule-based, 40% semantic)
    combined_score = 0.6 * rule_score + 0.4 * semantic_score
    print(f"\nProfile: {profile['name']}")
    print(f"Job Role: {job_role}")
    print(f"Required Skills: {required_skills}")
    print(f"Optional Skills: {optional_skills}")
    print(f"Profile Experience: {profile_experience}")
    print(f"Role Match Score: {role_match_score:.1f}%")
    print(f"Matching Required Skills: {matching_skills}")
    print(f"Matching Optional Skills: {matching_optional_skills}")
    print(f"Skills Match Percentage: {skills_match_percentage:.1f}%")
    print(f"Optional Skills Percentage: {optional_skills_percentage:.1f}%")
    print(f"Years of Experience (required {min_years}-{max_years if max_years else '∞'}): {years_experience}")
    print(f"Years Score: {years_score:.1f}%")
    print(f"Rule-based Score: {rule_score:.1f}%")
    print(f"Semantic Similarity Score: {semantic_score:.1f}%")
    print(f"Combined Score: {combined_score:.1f}%")
    # Determine rejection reason if not a match
    rejection_reason = ""
    if too_much_exp:
        if max_years > 0:
            rejection_reason = f"Too much experience ({years_experience} > {max_years})"
        else:
            rejection_reason = f"Too much experience (>{int(min_years * 1.2)})"
    elif combined_score <= 25:
        if len(matching_skills) == 0 and role_match_score == 0:
            rejection_reason = "No matching skills or relevant role experience"
        elif len(matching_skills) == 0:
            rejection_reason = "No matching required skills found"
        elif role_match_score == 0:
            rejection_reason = "No relevant role experience"
        elif years_score < 100:
            if years_experience < min_years:
                rejection_reason = f"Insufficient years of experience ({years_experience} < {min_years})"
            elif max_years and years_experience > max_years:
                rejection_reason = f"Too much experience ({years_experience} > {max_years})"
            elif min_years > 0 and years_experience > min_years * 1.2:
                rejection_reason = f"Too much experience (>{int(min_years * 1.2)})"
            else:
                rejection_reason = f"Years of experience not in range ({years_experience})"
        else:
            rejection_reason = f"Low combined score ({combined_score:.1f}% < 25%)"
    else:
        if len(matching_skills) == 0 and role_match_score == 0:
            rejection_reason = "No matching skills or relevant role experience"
        elif len(matching_skills) == 0:
            rejection_reason = "No matching required skills found"
        elif role_match_score == 0:
            rejection_reason = "No relevant role experience"
        elif years_score < 100:
            if years_experience < min_years:
                rejection_reason = f"Insufficient years of experience ({years_experience} < {min_years})"
            elif max_years and years_experience > max_years:
                rejection_reason = f"Too much experience ({years_experience} > {max_years})"
            elif min_years > 0 and years_experience > min_years * 1.2:
                rejection_reason = f"Too much experience (>{int(min_years * 1.2)})"
            else:
                rejection_reason = f"Years of experience not in range ({years_experience})"
    is_match = combined_score > 25 and (len(matching_skills) > 0 or role_match_score > 0) and years_score >= 80 and not too_much_exp
    if is_match:
        print("Profile matches job requirements!")
        rejection_reason = "N/A - Profile matched"
    else:
        print(f"Profile does not match job requirements. Reason: {rejection_reason}")
    # Add both semantic scores to profile for transparency
    profile['metadata']['st_semantic_score'] = st_semantic_score
    profile['metadata']['openai_semantic_score'] = openai_semantic_score
    return is_match, rule_score, semantic_score, combined_score, rejection_reason, matching_skills, matching_optional_skills, years_experience, 0

def process_zip_file(zip_path, output_dir, job_role=None, required_skills=None, optional_skills=None, min_years=0, max_years=0, location_type=0, onsite_location=""):
    """Process all PDF files in a ZIP archive and save all profiles with metadata. Also copy matching resumes to a separate folder."""
    try:
        if not os.path.exists(zip_path):
            logging.error(f"ZIP file not found at {zip_path}")
            return None
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")
        
        shortlisted_dir = os.path.join(output_dir, 'shortlisted_resumes')
        if not os.path.exists(shortlisted_dir):
            os.makedirs(shortlisted_dir)
            logging.info(f"Created shortlisted resumes directory: {shortlisted_dir}")
        
        to_be_approved_dir = os.path.join(output_dir, 'to_be_approved_by_hr')
        if not os.path.exists(to_be_approved_dir):
            os.makedirs(to_be_approved_dir)
            logging.info(f"Created to_be_approved_by_hr directory: {to_be_approved_dir}")
        
        all_profiles = []
        matching_profiles = []
        
        logging.info(f"Starting to process ZIP file: {zip_path}")
        start_time = datetime.now()
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            temp_dir = os.path.join(output_dir, 'temp_extract')
            logging.info(f"Extracting files to: {temp_dir}")
            zip_ref.extractall(temp_dir)
            
            pdf_count = 0
            processed_count = 0
            error_count = 0
            
            job_role_embedding = st_model.encode(job_role, convert_to_tensor=True) if job_role else None
            
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        pdf_count += 1
                        pdf_path = os.path.join(root, file)
                        logging.info(f"Processing resume {pdf_count}: {file}")
                        try:
                            profile = parse_resume(pdf_path, job_keywords=required_skills)
                            if profile:
                                processed_count += 1
                                # Add metadata
                                profile['metadata'] = {
                                    'filename': file,
                                    'processed_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'is_match': False,
                                    'resume_link': os.path.join(temp_dir, file)  # relative path to the extracted PDF
                                }
                                
                                # Compute required (matching) keywords
                                if required_skills:
                                    matching_keywords = set()
                                    for skill in required_skills:
                                        skill_lower = skill.lower().strip()
                                        for pk in profile.get('keywords', []):
                                            if fuzz.ratio(skill_lower, pk.lower()) > 85:
                                                matching_keywords.add(skill)
                                    profile['required_keywords'] = list(matching_keywords)
                                else:
                                    profile['required_keywords'] = []
                                
                                # Hybrid match
                                is_match, rule_score, semantic_score, combined_score, rejection_reason, matching_skills, matching_optional_skills, years_experience, _ = is_profile_match(
                                    profile, job_role, required_skills, optional_skills, min_years, max_years, location_type, onsite_location, job_role_embedding=job_role_embedding
                                )
                                profile['metadata']['rule_score'] = rule_score
                                profile['metadata']['semantic_score'] = semantic_score
                                profile['metadata']['combined_score'] = combined_score
                                profile['metadata']['rejection_reason'] = rejection_reason
                                
                                if is_match:
                                    logging.info(f"Found matching profile: {profile['name']}")
                                    profile['metadata']['is_match'] = True
                                    # Copy the matching resume PDF to shortlisted_resumes
                                    dest_path = os.path.join(shortlisted_dir, file)
                                    shutil.copy2(pdf_path, dest_path)
                                    # Update resume_link to point to the new location
                                    profile['metadata']['resume_link'] = dest_path
                                    matching_profiles.append(profile)
                                else:
                                    # Borderline: combined_score >20 and ≤25
                                    if combined_score >= 20 and combined_score <= 25:
                                        dest_path = os.path.join(to_be_approved_dir, file)
                                        shutil.copy2(pdf_path, dest_path)
                                        profile['metadata']['resume_link'] = dest_path
                                        profile['metadata']['needs_hr_approval'] = True
                                        profile['metadata']['rejection_reason'] = 'Needs approval by HR'
                                
                                all_profiles.append(profile)
                                save_resume_to_db(profile, job_role, required_skills, optional_skills)
                        except Exception as e:
                            error_count += 1
                            logging.error(f"Error processing {file}: {str(e)}")
            
            processing_time = datetime.now() - start_time
            logging.info(f"Processing completed in {processing_time}")
            logging.info(f"Statistics:")
            logging.info(f"- Total PDF files found: {pdf_count}")
            logging.info(f"- Successfully processed: {processed_count}")
            logging.info(f"- Errors encountered: {error_count}")
            logging.info(f"- Matching profiles found: {len(matching_profiles)}")
            
            # Clean up temporary directory
            for root, dirs, files in os.walk(temp_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(temp_dir)
            logging.info("Cleaned up temporary directory")
        
        # Save all profiles to a JSON file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_profiles_file = os.path.join(output_dir, f'all_profiles_{timestamp}.json')
        matching_profiles_file = os.path.join(output_dir, f'matching_profiles_{timestamp}.json')
        
        # Save all profiles
        with open(all_profiles_file, 'w', encoding='utf-8') as f:
            json.dump(all_profiles, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved all profiles to: {all_profiles_file}")
        
        # Save matching profiles
        with open(matching_profiles_file, 'w', encoding='utf-8') as f:
            json.dump(matching_profiles, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved matching profiles to: {matching_profiles_file}")
        
        return all_profiles_file, matching_profiles_file
        
    except Exception as e:
        logging.error(f"Error processing ZIP file: {str(e)}", exc_info=True)
        return None

def anonymize_resume_text(text, name=None):
    """Remove/mask personal info (name, email, phone, LinkedIn, GitHub, address, etc.) from resume text."""
    # Mask emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    # Mask phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
    text = re.sub(r'\b\d{10}\b', '[PHONE]', text)
    # Mask LinkedIn/GitHub URLs
    text = re.sub(r'https?://(www\.)?(linkedin|github)\.com/\S+', '[LINK]', text, flags=re.IGNORECASE)
    # Mask addresses (simple heuristic: lines with numbers and street/road/avenue)
    text = re.sub(r'\d+\s+\w+\s+(Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Block|Sector|Colony|Apartment|Apt|Suite|Floor|Building|Bldg|House|Flat)', '[ADDRESS]', text, flags=re.IGNORECASE)
    # Mask names using spaCy NER if provided
    if name and name != "Unknown":
        text = text.replace(name, '[NAME]')
    else:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                text = text.replace(ent.text, '[NAME]')
    return text

def get_openai_certification_and_structure_score(anonymized_text, openai_api_key=None):
    """Call OpenAI API to extract certifications and rate CV structure. Returns (certifications, structure_score, comments)."""
    try:
        import openai
        import importlib.metadata
    except ImportError:
        return [], 0, "OpenAI package not installed"
    if openai_api_key is None:
        openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        return [], 0, "OpenAI API key not set"
    openai.api_key = "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"
    prompt = (
        "You are an expert resume reviewer. Given the following anonymized resume text, "
        "1. List all professional certifications found (as a Python list). "
        "2. Rate the structure and clarity of the CV on a scale of 1-10 (10 = excellent structure, 1 = poor structure). "
        "3. Provide a brief comment on the structure.\n\n"
        "Resume:\n" + anonymized_text + "\n\n"
        "Respond in the following JSON format: {\"certifications\": [...], \"structure_score\": <int>, \"comment\": \"...\"}"
    )
    try:
        try:
            version = importlib.metadata.version('openai')
        except Exception:
            version = '0.0.0'
        major_version = int(version.split('.')[0])
        if major_version >= 1:
            # OpenAI v1.x syntax
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.2
            )
            content = response.choices[0].message.content
        else:
            return [], 0, "OpenAI v0.x is not supported. Please upgrade the openai package."
        # Find JSON in response
        try:
            if isinstance(content, str):
                result = json.loads(content)
            else:
                return [], 0, "OpenAI response content is not a string."
        except Exception:
            # Try to extract JSON substring
            import re
            if isinstance(content, str):
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    result = json.loads(match.group(0))
                else:
                    return [], 0, "Could not parse OpenAI response"
            else:
                return [], 0, "OpenAI response content is not a string."
        certifications = result.get('certifications', [])
        structure_score = result.get('structure_score', 0)
        comment = result.get('comment', '')
        return certifications, structure_score, comment
    except Exception as e:
        return [], 0, f"OpenAI API error: {str(e)}"

if __name__ == "__main__":
    try:
        # Setup logging
        log_file = setup_logging()
        logging.info("Starting resume parser")
        # Get job description from user
        print("\nEnter the job description:")
        job_description = input().strip()
        logging.info(f"Job description received: {job_description}")
        # Get job role from user
        print("\nEnter the job role (e.g., Accountant, Software Engineer, Marketing Manager):")
        job_role = input().strip()
        logging.info(f"Job role received: {job_role}")
        # Get required skills from user
        print("\nEnter the required skills (comma-separated, e.g., QuickBooks, GAAP, Excel):")
        skills_input = input().strip()
        required_skills = [skill.strip() for skill in skills_input.split(',') if skill.strip()]
        logging.info(f"Required skills received: {required_skills}")
        # Get optional skills from user
        print("\nEnter the optional skills (comma-separated, or leave blank):")
        optional_skills_input = input().strip()
        optional_skills = [skill.strip() for skill in optional_skills_input.split(',') if skill.strip()]
        logging.info(f"Optional skills received: {optional_skills}")
        # Get minimum years of experience required
        print("\nEnter the minimum years of experience required:")
        min_years = int(input().strip())
        logging.info(f"Minimum years of experience required: {min_years}")
        # Get maximum years of experience allowed
        print("\nEnter the maximum years of experience allowed (0 for no max):")
        max_years = int(input().strip())
        logging.info(f"Maximum years of experience allowed: {max_years}")
        # Get location type
        print("\nEnter 1 for onsite, 0 for remote:")
        location_type = int(input().strip())
        onsite_location = ""
        if location_type == 1:
            print("\nEnter the onsite location (city/state/country):")
            onsite_location = input().strip()
            logging.info(f"Onsite location: {onsite_location}")
        else:
            logging.info("Remote position")
        # Process resumes
        zip_path = "resumes.zip"
        output_dir = "output"
        logging.info("Starting resume processing...")
        output_files = process_zip_file(zip_path, output_dir, job_role, required_skills, optional_skills, min_years, max_years, location_type, onsite_location)
        if output_files:
            all_profiles_file, matching_profiles_file = output_files
            logging.info("Processing completed successfully")
            print(f"\nAll profiles saved to: {all_profiles_file}")
            print(f"Matching profiles saved to: {matching_profiles_file}")
            print(f"Log file: {log_file}")
        else:
            logging.error("Processing failed")
            print("\nAn error occurred while processing the resumes.")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        print(f"\nAn error occurred: {str(e)}")          