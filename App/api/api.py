from flask import Flask, flash, request, redirect, url_for, send_from_directory, session
import time
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger('HELLO WORLD')

app = Flask(__name__, static_folder="../build", static_url_path='/')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = './uploads'

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

# @app.route('/upload', methods = ["GET", "POST"])
# def upload():
#         if request.method == "POST":
#             #ngecek apakah dia POST
#             if 'file' not in request.files:
#                 flash('Tidak ada file')
#                 return redirect(request.url)
#             file = request.files['file']
#             if file.filename == '':
#                 flash('Tidak ada file')
#                 return redirect(request.url)
#             if file and allowed_file(file.filename):
#                 filename = secure_filename(file.filename)

#                 # isi fungsi untuk manipulasi file disini

#                 file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#                 return redirect(url_for('download_file', name=filename))

@app.route('../uploads/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)

@app.route('/upload', methods=['POST'])
def fileUpload():
    target=os.path.join(UPLOAD_FOLDER,'test_docs')
    if not os.path.isdir(target):
        os.mkdir(target)
    logger.info("welcome to upload`")
    file = request.files['file'] 
    filename = secure_filename(file.filename)
    destination="/".join([target, filename])
    file.save(destination)
    session['uploadFilePath']=destination
    response="Whatever you wish too return"
    return response

if __name__ == "__main__":
    app.secret_key = os.urandom(24)
    app.run(debug=True,host="0.0.0.0",use_reloader=False)

flask_cors.CORS(app, expose_headers='Authorization')