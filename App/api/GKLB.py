import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.image import imread
import numpy as np
import os

from numpy.linalg import svd

plt.rcParams['figure.figsize'] = [16,6]
plt.rcParams.update({'font.size': 18})

# Define randomized SVD function
def rSVD(X,r,q,p):
    # Step 1: Sample column space of X with P matrix
    ny = X.shape[1]
    P = np.random.randn(ny,r+p)
    Z = X @ P
    for k in range(q):
        Z = X @ (X.T @ Z)

    Q, R = np.linalg.qr(Z,mode='reduced')

    # Step 2: Compute SVD on projected Y = Q.T @ X
    Y = Q.T @ X
    UY, S, VT = np.linalg.svd(Y)
    U = Q @ UY

    return U, S, VT

img = np.asarray(Image.open(r'C:\Users\irfan\OneDrive\Documents\Nando\Informatika\tubes2_algeo\Algeo02-20007\App\api\jupiter.jpg'))
k = 5

r = img[:,:,0]  # array for R
g = img[:,:,1]  # array for G
b = img[:,:,2] # array for B
    
print("compressing...")
    
    # Calculating the svd components for all three arrays
ur,sr,vr = svd(r, full_matrices=False)
ug,sg,vg = svd(g, full_matrices=False)
ub,sb,vb = svd(b, full_matrices=False)
    
    # Forming the compress image with reduced information
    # We are selecting only k singular values for each array to make image which will exclude some information from the 
    # image while image will be of same dimension
    
    # ur (mxk), diag(sr) (kxk) and vr (kxn) if image is off (mxn)
    # so let suppose we only selecting the k1 singular value from diag(sr) to form image
    
rr = np.dot(ur[:,:k],np.dot(np.diag(sr[:k]), vr[:k,:]))
rg = np.dot(ug[:,:k],np.dot(np.diag(sg[:k]), vg[:k,:]))
rb = np.dot(ub[:,:k],np.dot(np.diag(sb[:k]), vb[:k,:]))
    
print("arranging...")
    
    # Creating a array of zeroes; shape will be same as of image matrix
rimg = np.zeros(img.shape)
    
    # Adding matrix for R, G & B in created array
rimg[:,:,0] = rr
rimg[:,:,1] = rg
rimg[:,:,2] = rb
    
    # It will check if any value will be less than 0 will be converted to its absolute
    # and, if any value is greater than 255 than it will be converted to 255
    # because in image array of unit8 can only have value between 0 & 255
for ind1, row in enumerate(rimg):
    for ind2, col in enumerate(row):
        for ind3, value in enumerate(col):
            if value < 0:
                rimg[ind1,ind2,ind3] = abs(value)
            if value > 255:
                rimg[ind1,ind2,ind3] = 255

    # converting the compress image array to uint8 type for further conversion into image object
compressed_image = rimg.astype(np.uint8)
    
    # Showing the compressed image in graph

    
    # Uncomment below code if you want to save your compressed image to the file
compressed_image = Image.fromarray(compressed_image)
compressed_image.save("another.jpg")
#rU, rS, rVT = rSVD(X,r,q,p)

# XSVD = U[:,:(r+1)] @ np.diag(S[:(r+1)]) @ VT[:(r+1),:] # SVD approximation
# errSVD = np.linalg.norm(X-XSVD,ord=2) / np.linalg.norm(X,ord=2)

#XrSVD = rU[:,:(r+1)] @ np.diag(rS[:(r+1)]) @ rVT[:(r+1),:] # SVD approximation
#errSVD = np.linalg.norm(X-XrSVD,ord=2) / np.linalg.norm(X,ord=2)

# img1 = Image.fromarray(XSVD, 'RGB')
# #img2 = Image.fromarray(XrSVD, 'RGB')

# img1.save('img1.jpg')
# #img2.save('img2.jpg')
# print("hello world")
