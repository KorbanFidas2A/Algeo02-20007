from flask import Flask, flash, request, redirect, url_for, send_from_directory
import time
import os
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder="../build", static_url_path='/')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = '../uploads'

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/time')
def get_current_time():
    return {'time' : time.time()}
    #jasonified

@app.route('/upload', methods = ["GET", "POST"])
def upload():
        if request.method == "POST":
            #ngecek apakah dia POST
            if 'file' not in request.files:
                flash('Tidak ada file')
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                flash('Tidak ada file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                return redirect(url_for('download_file', name=filename))

@app.route('../uploads/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)
