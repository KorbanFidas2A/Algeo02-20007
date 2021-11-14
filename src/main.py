from PIL import Image
import numpy as np
from numpy.linalg import norm
from math import sqrt
from random import normalvariate


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
    ny = X.shape[1]
    P = np.random.randn(ny,r+p)
    Z = X @ P
    for k in range(q):
        Z = X @ (X.T @ Z)

    Q, R = np.linalg.qr(Z,mode='reduced')

    Y = Q.T @ X
    UY, S, VT = svd(Y)
    U = Q @ UY

    return U, S, VT


img = np.asarray(Image.open(r'jupiter.jpg'))
k = int(input("compression percentage: "))
m = img.shape[0]
n = img.shape[1]

# array RGB
r = img[:,:,0]
g = img[:,:,1]
b = img[:,:,2]

print("compressing...")
    
q = 1
p = 5

Ur, Sr, VTr = randomSVD(r,k,q,p)
Ug, Sg, VTg = randomSVD(g,k,q,p)
Ub, Sb, VTb = randomSVD(b,k,q,p)

# singular values yang digunakan sesuai persentase yang diinginkan
k = round((k*m*n)/(100*(m+1+n)))

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
compressed_image.save("../test/anotherxxx.jpg")