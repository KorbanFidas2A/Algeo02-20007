from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
import urllib.request
import os
from werkzeug.utils import secure_filename

from PIL import Image
import numpy as np
from numpy.linalg import norm
from math import sqrt
from random import normalvariate
import time

 
app = Flask(__name__)
 
UPLOAD_FOLDER = 'D:/algeodum/api/uploads'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def randomUnitVector(n):
    unnormalized = [normalvariate(0, 1) for _ in range(n)]
    theNorm = sqrt(sum(x * x for x in unnormalized))
    return [x / theNorm for x in unnormalized]


def svd_1D(A, epsilon=1e-10):
    n, m = A.shape
    x = randomUnitVector(min(n,m))
    lastV = None
    currentV = x

    if n > m:
        B = np.dot(A.T, A)
    else:
        B = np.dot(A, A.T)

    iterations = 0
    #lakukan iterasi sampai dengan nilai vektor hasil iterasi terakhir - iterasi sebelumnya 
    # mencapai epsilon (sudah konvergen ke suatu nilai, i.e  eigenvektor terbesar)
    while True:
        iterations += 1
        lastV = currentV
        currentV = np.dot(B, lastV)
        currentV = currentV / norm(currentV)

        if abs(np.dot(currentV, lastV)) > 1 - epsilon:
            return currentV


def svd(A, k=None, epsilon=1e-10):
    A = np.array(A, dtype=float)
    n, m = A.shape
    svdSoFar = []
    if k is None:
        k = min(n, m)

    for i in range(k):
        matrixFor1D = A.copy()

        for singularValue, u, v in svdSoFar[:i]:
            matrixFor1D -= singularValue * np.outer(u, v)

        if n > m:
            v = svd_1D(matrixFor1D, epsilon=epsilon)  # vektor singular selanjutnya
            u_unnormalized = np.dot(A, v)
            sigma = norm(u_unnormalized)  # singular value selanjutnya
            u = u_unnormalized / sigma
        else:
            u = svd_1D(matrixFor1D, epsilon=epsilon)  # vektor singular selanjutnya
            v_unnormalized = np.dot(A.T, u)
            sigma = norm(v_unnormalized)  # singular value selanjutnya
            v = v_unnormalized / sigma

        svdSoFar.append((sigma, u, v))

    singularValues, us, vs = [np.array(x) for x in zip(*svdSoFar)]
    return us.T, singularValues, vs


def randomSVD(X,r,q,p):
    #FASE 1: dekomposisi matrix X dengan cara sampling kolom X dengan random oleh matrix P
    ny = X.shape[1]
    P = np.random.randn(ny,r+p)
    Z = X @ P
    for k in range(q):
        Z = X @ (X.T @ Z)

    Q, R = np.linalg.qr(Z,mode='reduced')
    #FASE 2: Menghitung SVD dari matrix Y
    Y = Q.T @ X
    UY, S, VT = svd(Y)
    U = Q @ UY

    return U, S, VT

def imgcompres(x, k):
    img = np.asarray(Image.open(x))
    m = img.shape[0]
    n = img.shape[1]

    # array RGB
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]

    print("compressing...")
    starttime =  time.time()
        
    q = 1
    p = 5



    Ur, Sr, VTr = randomSVD(r,k,q,p)
    Ug, Sg, VTg = randomSVD(g,k,q,p)
    Ub, Sb, VTb = randomSVD(b,k,q,p)


    k = round(((100-k)*m*n)/(100*(m+1+n)))


    # menggabungkan tiap komponen svd sebanyak k kolom/baris
    rr = np.dot(Ur[:,:k],np.dot(np.diag(Sr[:k]), VTr[:k,:]))
    rg = np.dot(Ug[:,:k],np.dot(np.diag(Sg[:k]), VTg[:k,:]))
    rb = np.dot(Ub[:,:k],np.dot(np.diag(Sb[:k]), VTb[:k,:]))
        
    print("arranging...")

    # buat array kosong untuk menampung matriks gambar kompresi 
    rimg = np.zeros(img.shape)

    # menggabungkan tiap channel RGB
    rimg[:,:,0] = rr
    rimg[:,:,1] = rg
    rimg[:,:,2] = rb

    # konversi angka matriks  
    for ind1, row in enumerate(rimg):
        for ind2, col in enumerate(row):
            for ind3, value in enumerate(col):
                if value < 0:
                    rimg[ind1,ind2,ind3] = abs(value)
                if value > 255:
                    rimg[ind1,ind2,ind3] = 255

    # konversi gambar & save file
    compressed_image = rimg.astype(np.uint8)
    compressed_image = Image.fromarray(compressed_image)
    #NAD KALO MAU GANTI PATH SAVE DISINI YA

    end = time.time()

    # total time taken
    print(f"Runtime of the program is {end - starttime}")

    return compressed_image
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if 'file' not in request.files:
        flash('Tidak ada file.')
        return redirect(request.url)
    file = request.files['file']
    constant = request.form.get('cons', type=int)
    if file.filename == '':
        flash('Tidak ada foto yang Anda pilih.')
        return redirect(request.url)
    if constant == 0:
        flash('Masukkan angka yang benar, yaitu lebih dari 0.')
        return redirect(request.url)
    if file and allowed_file(file.filename):

        # pemrosesan file dilakukan disini
        
        #penghitungan waktu
        starttime =  time.time()

        #file extension
        if(file.filename.rsplit('.', 1)[1].lower() == 'jpg'):
            file = imgcompres(file.filename, constant)
            filename = "result.jpg"
        elif(file.filename.rsplit('.', 1)[1].lower() == 'png'):
            file = imgcompres(file.filename, constant)
            filename = "result.png"
        else:
            file = imgcompres(file.filename, constant)
            filename = "result.jpeg"

        end = time.time()

        flash(f"Image compression time is {end - starttime} s")
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Foto Anda telah berhasil di upload')
        return redirect(url_for('download_file', name=filename))
    else:
        flash('Anda hanya boleh mengupload jpg, jpeg atau png.')
        return redirect(request.url)

@app.route('/uploads/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)

 
if __name__ == "__main__":
    app.run()
