from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, send_file
import os
import shutil
import zipfile
from resume_parser import process_zip_file
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Needed for flashing messages
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
ALLOWED_EXTENSIONS = {'zip'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        if 'resumes_zip' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['resumes_zip']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        # Get user-specified zip filename or default
        zip_filename = request.form.get('zip_filename', '').strip()
        if not zip_filename:
            zip_filename = 'resumes.zip'
        if not zip_filename.lower().endswith('.zip'):
            zip_filename += '.zip'
        if file and allowed_file(file.filename):
            zip_path = os.path.join(UPLOAD_FOLDER, zip_filename)
            file.save(zip_path)
        else:
            flash('Invalid file type. Please upload a .zip file.')
            return redirect(request.url)
        # Get form data
        job_description = request.form.get('job_description', '')
        job_role = request.form.get('job_role', '')
        required_skills = [s.strip() for s in request.form.get('required_skills', '').split(',') if s.strip()]
        optional_skills = [s.strip() for s in request.form.get('optional_skills', '').split(',') if s.strip()]
        min_years = int(request.form.get('min_years', '0'))
        max_years = int(request.form.get('max_years', '0'))
        location_type = int(request.form.get('location_type', '0'))
        onsite_location = request.form.get('onsite_location', '') if location_type == 1 else ''
        # Run the parser
        # Create a unique output folder per job role and timestamp
        safe_job_role = ''.join(c if c.isalnum() or c in (' ', '_') else '_' for c in job_role).replace(' ', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        job_output_folder = f"output_{safe_job_role}_{timestamp}"
        job_output_path = os.path.join(OUTPUT_FOLDER, job_output_folder)
        if not os.path.exists(job_output_path):
            os.makedirs(job_output_path)
        output_files = process_zip_file(zip_path, job_output_path, job_role, required_skills, optional_skills, min_years, max_years, 0, "")
        if output_files:
            all_profiles_file, matching_profiles_file = output_files
            # Prepare links to output files and folders
            shortlisted_dir = os.path.join(job_output_path, 'shortlisted_resumes')
            to_be_approved_dir = os.path.join(job_output_path, 'to_be_approved_by_hr')
            return render_template('results.html',
                                   all_profiles_file=all_profiles_file,
                                   matching_profiles_file=matching_profiles_file,
                                   shortlisted_dir=shortlisted_dir,
                                   to_be_approved_dir=to_be_approved_dir,
                                   job_folder=job_output_folder)
        else:
            flash('An error occurred while processing the resumes.')
            return redirect(request.url)
    return render_template('index.html')

@app.route('/results/<job_folder>')
def results(job_folder):
    job_output_path = os.path.join(OUTPUT_FOLDER, job_folder)
    all_profiles_file = None
    matching_profiles_file = None
    # Find the latest all_profiles_*.json and matching_profiles_*.json in the folder
    if os.path.exists(job_output_path):
        files = os.listdir(job_output_path)
        all_profiles = [f for f in files if f.startswith('all_profiles_') and f.endswith('.json')]
        matching_profiles = [f for f in files if f.startswith('matching_profiles_') and f.endswith('.json')]
        if all_profiles:
            all_profiles.sort(reverse=True)
            all_profiles_file = os.path.join(job_output_path, all_profiles[0])
        if matching_profiles:
            matching_profiles.sort(reverse=True)
            matching_profiles_file = os.path.join(job_output_path, matching_profiles[0])
    shortlisted_dir = os.path.join(job_output_path, 'shortlisted_resumes')
    to_be_approved_dir = os.path.join(job_output_path, 'to_be_approved_by_hr')
    return render_template('results.html',
                           all_profiles_file=all_profiles_file,
                           matching_profiles_file=matching_profiles_file,
                           shortlisted_dir=shortlisted_dir,
                           to_be_approved_dir=to_be_approved_dir,
                           job_folder=job_folder)

@app.route('/download/<path:filename>')
def download_file(filename):
    directory = os.path.dirname(filename)
    fname = os.path.basename(filename)
    return send_from_directory(directory, fname, as_attachment=True)

@app.route('/browse/<path:folder>')
def browse_folder(folder):
    folder_path = os.path.join(OUTPUT_FOLDER, folder)
    files = os.listdir(folder_path) if os.path.exists(folder_path) else []
    return render_template('browse.html', folder=folder, files=files)

@app.route('/download_shortlisted_zip/<job_folder>')
def download_shortlisted_zip(job_folder):
    shortlisted_dir = os.path.join(OUTPUT_FOLDER, job_folder, 'shortlisted_resumes')
    zip_path = os.path.join(OUTPUT_FOLDER, job_folder, 'shortlisted_resumes.zip')
    # Create ZIP of all PDFs in shortlisted_dir
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, _, files in os.walk(shortlisted_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, shortlisted_dir)
                    zipf.write(file_path, arcname)
    return send_file(zip_path, as_attachment=True)

@app.route('/download_to_be_approved_zip/<job_folder>')
def download_to_be_approved_zip(job_folder):
    to_be_approved_dir = os.path.join(OUTPUT_FOLDER, job_folder, 'to_be_approved_by_hr')
    zip_path = os.path.join(OUTPUT_FOLDER, job_folder, 'to_be_approved_by_hr.zip')
    # Create ZIP of all PDFs in to_be_approved_dir
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, _, files in os.walk(to_be_approved_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, to_be_approved_dir)
                    zipf.write(file_path, arcname)
    return send_file(zip_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True) 