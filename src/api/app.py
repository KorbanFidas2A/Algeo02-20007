from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
import urllib.request
import os
from werkzeug.utils import secure_filename
 
app = Flask(__name__)
 
UPLOAD_FOLDER = 'D:/algeodum/api/uploads'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('Tidak ada file.')
        return redirect(request.url)
    file = request.files['file']
    constant = request.form.get('cons', type=int)
    if file.filename == '':
        flash('Tidak ada foto yang Anda pilih.')
        return redirect(request.url)
    if constant <= 0:
        flash('Masukkan angka yang benar, yaitu lebih dari 0.')
        return redirect(request.url)
    if file and allowed_file(file.filename):

        # pemrosesan file dilakukan disini


        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Foto Anda telah berhasil di upload')
        return redirect(url_for('download_file', name=filename))
    else:
        flash('Anda hanya boleh mengupload jpg, jpeg atau png.')
        return redirect(request.url)
 
if __name__ == "__main__":
    app.run()
