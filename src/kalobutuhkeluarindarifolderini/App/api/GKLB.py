import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.image import imread
import numpy as np
import os
import copy
import math
from math import sqrt
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
    #UY, S, VT = np.linalg.svd(Y)
    UY, S, VT = svd(Y)
    U = Q @ UY

    return U, S, VT

def svdA(A, k, epsilon=0.00001):
    #source http://mlwiki.org/index.php/Power_Iteration
    #adjusted to work with n<m and n>m matrices
    n_orig, m_orig = A.shape
    if k is None:
        k=min(n_orig,m_orig)
        
    A_orig=A.copy()
    if n_orig > m_orig:
        A = A.T @ A
        n, m = A.shape
    elif n_orig < m_orig:
        A = A @ A.T
        n, m = A.shape
    else:
        n,m=n_orig, m_orig
        
    Q = np.random.rand(n, k)
    Q, _ = np.linalg.qr(Q)
    Q_prev = Q
 
    for i in range(1000):
        Z = A @ Q
        Q, R = np.linalg.qr(Z)
        # can use other stopping criteria as well 
        err = ((Q - Q_prev) ** 2).sum()
        Q_prev = Q
        if err < epsilon:
            break
            
    singular_values=np.sqrt(np.diag(R))    
    if n_orig < m_orig: 
        left_vecs=Q.T
        #use property Values @ V = U.T@A => V=inv(Values)@U.T@A
        right_vecs=np.linalg.inv(np.diag(singular_values))@left_vecs.T@A_orig
    elif n_orig==m_orig:
        left_vecs=Q.T
        right_vecs=left_vecs
        singular_values=np.square(singular_values)
    else:
        right_vecs=Q.T
        #use property Values @ V = U.T@A => U=A@V@inv(Values)
        left_vecs=A_orig@ right_vecs.T @np.linalg.inv(np.diag(singular_values))

    return left_vecs, singular_values, right_vecs

def svdB(a):
    '''Compute the singular value decomposition of array.'''

    # Golub and Reinsch state that eps should not be smaller than the
    # machine precision, ie the smallest number
    # for which 1+e>1.  tol should be beta/e where beta is the smallest
    # positive number representable in the computer.
    eps = 1.e-15  # assumes double precision
    tol = 1.e-64/eps
    assert 1.0+eps > 1.0 # if this fails, make eps bigger
    assert tol > 0.0     # if this fails, make tol bigger
    itmax = 50
    u = copy.deepcopy(a)
    m = len(a)
    n = len(a[0])
    #if __debug__: print 'a is ',m,' by ',n

    if m < n:
        if __debug__: print ('Error: m is less than n')
        raise (ValueError,'SVD Error: m is less than n.')

    e = [0.0]*n  # allocate arrays
    q = [0.0]*n
    v = []
    for k in range(n): v.append([0.0]*n)
 
    # Householder's reduction to bidiagonal form

    g = 0.0
    x = 0.0

    for i in range(n):
        e[i] = g
        s = 0.0
        l = i+1
        for j in range(i,m): s += (u[j][i]*u[j][i])
        if s <= tol:
            g = 0.0
        else:
            f = u[i][i]
            if f < 0.0:
                g = math.sqrt(s)
            else:
                g = -math.sqrt(s)
            h = f*g-s
            u[i][i] = f-g
            for j in range(l,n):
                s = 0.0
                for k in range(i,m): s += u[k][i]*u[k][j]
                f = s/h
                for k in range(i,m): u[k][j] = u[k][j] + f*u[k][i]
        q[i] = g
        s = 0.0
        for j in range(l,n): s = s + u[i][j]*u[i][j]
        if s <= tol:
            g = 0.0
        else:
            f = u[i][i+1]
            if f < 0.0:
                g = math.sqrt(s)
            else:
                g = -math.sqrt(s)
            h = f*g - s
            u[i][i+1] = f-g
            for j in range(l,n): e[j] = u[i][j]/h
            for j in range(l,m):
                s=0.0
                for k in range(l,n): s = s+(u[j][k]*u[i][k])
                for k in range(l,n): u[j][k] = u[j][k]+(s*e[k])
        y = abs(q[i])+abs(e[i])
        if y>x: x=y
    # accumulation of right hand gtransformations
    for i in range(n-1,-1,-1):
        if g != 0.0:
            h = g*u[i][i+1]
            for j in range(l,n): v[j][i] = u[i][j]/h
            for j in range(l,n):
                s=0.0
                for k in range(l,n): s += (u[i][k]*v[k][j])
                for k in range(l,n): v[k][j] += (s*v[k][i])
        for j in range(l,n):
            v[i][j] = 0.0
            v[j][i] = 0.0
        v[i][i] = 1.0
        g = e[i]
        l = i
    #accumulation of left hand transformations
    for i in range(n-1,-1,-1):
        l = i+1
        g = q[i]
        for j in range(l,n): u[i][j] = 0.0
        if g != 0.0:
            h = u[i][i]*g
            for j in range(l,n):
                s=0.0
                for k in range(l,m): s += (u[k][i]*u[k][j])
                f = s/h
                for k in range(i,m): u[k][j] += (f*u[k][i])
            for j in range(i,m): u[j][i] = u[j][i]/g
        else:
            for j in range(i,m): u[j][i] = 0.0
        u[i][i] += 1.0
    #diagonalization of the bidiagonal form
    eps = eps*x
    for k in range(n-1,-1,-1):
        for iteration in range(itmax):
            # test f splitting
            for l in range(k,-1,-1):
                goto_test_f_convergence = False
                if abs(e[l]) <= eps:
                    # goto test f convergence
                    goto_test_f_convergence = True
                    break  # break out of l loop
                if abs(q[l-1]) <= eps:
                    # goto cancellation
                    break  # break out of l loop
            if not goto_test_f_convergence:
                #cancellation of e[l] if l>0
                c = 0.0
                s = 1.0
                l1 = l-1
                for i in range(l,k+1):
                    f = s*e[i]
                    e[i] = c*e[i]
                    if abs(f) <= eps:
                        #goto test f convergence
                        break
                    g = q[i]
                    h = pythag(f,g)
                    q[i] = h
                    c = g/h
                    s = -f/h
                    for j in range(m):
                        y = u[j][l1]
                        z = u[j][i]
                        u[j][l1] = y*c+z*s
                        u[j][i] = -y*s+z*c
            # test f convergence
            z = q[k]
            if l == k:
                # convergence
                if z<0.0:
                    #q[k] is made non-negative
                    q[k] = -z
                    for j in range(n):
                        v[j][k] = -v[j][k]
                break  # break out of iteration loop and move on to next k value
            if iteration >= itmax-1:
                if __debug__: print ('Error: no convergence.')
                # should this move on the the next k or exit with error??
                #raise ValueError,'SVD Error: No convergence.'  # exit the program with error
                break  # break out of iteration loop and move on to next k
            # shift from bottom 2x2 minor
            x = q[l]
            y = q[k-1]
            g = e[k-1]
            h = e[k]
            f = ((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y)
            g = pythag(f,1.0)
            if f < 0:
                f = ((x-z)*(x+z)+h*(y/(f-g)-h))/x
            else:
                f = ((x-z)*(x+z)+h*(y/(f+g)-h))/x
            # next QR transformation
            c = 1.0
            s = 1.0
            for i in range(l+1,k+1):
                g = e[i]
                y = q[i]
                h = s*g
                g = c*g
                z = pythag(f,h)
                e[i-1] = z
                c = f/z
                s = h/z
                f = x*c+g*s
                g = -x*s+g*c
                h = y*s
                y = y*c
                for j in range(n):
                    x = v[j][i-1]
                    z = v[j][i]
                    v[j][i-1] = x*c+z*s
                    v[j][i] = -x*s+z*c
                z = pythag(f,h)
                q[i-1] = z
                c = f/z
                s = h/z
                f = c*g+s*y
                x = -s*g+c*y
                for j in range(m):
                    y = u[j][i-1]
                    z = u[j][i]
                    u[j][i-1] = y*c+z*s
                    u[j][i] = -y*s+z*c
            e[l] = 0.0
            e[k] = f
            q[k] = x
            # goto test f splitting
        
            
    #vt = transpose(v)
    #return (u,q,vt)
    return (u,q,v)

def pythag(a,b):
    absa = abs(a)
    absb = abs(b)
    if absa > absb: return absa*math.sqrt(1.0+(absb/absa)**2)
    else:
        if absb == 0.0: return 0.0
        else: return absb*math.sqrt(1.0+(absa/absb)**2)

def transpose(a):
    '''Compute the transpose of a matrix.'''
    m = len(a)
    n = len(a[0])
    at = []
    for i in range(n): at.append([0.0]*m)
    for i in range(m):
        for j in range(n):
            at[j][i]=a[i][j]
    return at

def matrixmultiply(a,b):
    '''Multiply two matrices.
    a must be two dimensional
    b can be one or two dimensional.'''
    
    am = len(a)
    bm = len(b)
    an = len(a[0])
    try:
        bn = len(b[0])
    except TypeError:
        bn = 1
    if an != bm:
        raise (ValueError, 'matrixmultiply error: array sizes do not match.')
    cm = am
    cn = bn
    if bn == 1:
        c = [0.0]*cm
    else:
        c = []
        for k in range(cm): c.append([0.0]*cn)
    for i in range(cm):
        for j in range(cn):
            for k in range(an):
                if bn == 1:
                    c[i] += a[i][k]*b[k]
                else:
                    c[i][j] += a[i][k]*b[k][j]
    
    return c

def eigenvalue(A, v):
    val = A @ v / v
    return val[0]

def svd_dominant_eigen(A, epsilon=0.01):
    """returns dominant eigenvalue and dominant eigenvector of matrix A"""
    n, m = A.shape
    k=min(n,m)
    v = np.ones(k) / np.sqrt(k)
    if n > m:
        A = A.T @ A
    elif n < m:
        A = A @ A.T
    
    ev = eigenvalue(A, v)

    while True:
        Av = A@ v
        v_new = Av / np.linalg.norm(Av)
        ev_new = eigenvalue(A, v_new)
        if np.abs(ev - ev_new) < epsilon:
            break

        v = v_new
        ev = ev_new

    return ev_new, v_new

def svdC(A, k=None, epsilon=1e-10):
    """returns k dominant eigenvalues and eigenvectors of matrix A"""
    A = np.array(A, dtype=float)
    n, m = A.shape
        
    svd_so_far = []
    if k is None:
        k = min(n, m)

    for i in range(k):
        matrix_for_1d = A.copy()

        for singular_value, u, v in svd_so_far[:i]:
            matrix_for_1d -= singular_value * np.outer(u, v)

        if n > m:
            _, v = svd_dominant_eigen(matrix_for_1d, epsilon=epsilon)  # next singular vector
            u_unnormalized = A @ v
            sigma = np.linalg.norm(u_unnormalized)  # next singular value
            u = u_unnormalized / sigma
        else:
            _, u = svd_dominant_eigen(matrix_for_1d, epsilon=epsilon)  # next singular vector
            v_unnormalized = A.T @ u
            sigma = np.linalg.norm(v_unnormalized)  # next singular value
            v = v_unnormalized / sigma

        svd_so_far.append((sigma, u, v))

    singular_values, us, vs = [np.array(x) for x in zip(*svd_so_far)]
    return singular_values, us, vs

def svdD(A):
    u = A @ A.T
    left_eig_val, left_eig_vec = np.linalg.eigh(u)
    print(left_eig_val)
    singular_val = np.sqrt(left_eig_val)
    initial_k = len(singular_val)
    left_vec = np.linalg.norm(left_eig_vec)
    sigma = np.diag(singular_val)
    v = A.T @ A
    right_eig_val, right_eig_vec = np.linalg.eigh(v)
    right_vec = np.linalg.norm(right_eig_vec)

    return left_vec, sigma, right_vec, initial_k
    
def compute_sigma(evalues, m, n):
    """
    Compute sigma(middle) matrix of SVD
    :param evalues:
    :param m:
    :param n:
    :return:
    """
    sigma = np.zeros((m, n))

    for i in range(m):
        try:
            sigma[i, i] = evalues[i] ** 0.5
        except IndexError as e:
            continue

    return sigma


def compute_V(evalues, evectors):
    """
    Compute V(right side matrix) of SVD
    :param evalues:
    :param evectors:
    :return:
    """
    evectors = evectors.T

    evalues, evectors = zip(*sorted(zip(evalues, evectors), reverse=True))
    evectors = np.array(evectors)

    V = evectors.T

    return V


def compute_U(matrix, S, V, n):
    """
    Compute U(left side matrix) of SVD
    :param matrix:
    :param S:
    :param V:
    :param n:
    :return:
    """
    UT = np.zeros((len(S), len(matrix)))

    n, m = S.shape

    for i in range(min(n, m)):
        d = np.dot((1 / S[i, i]), matrix)
        UT[i] = np.dot(d, V[i])

    U = UT.T

    return U


def svdE(matrix):
    """
    SVD decomposition algorithm
    Decompose a given matrix to 3 matrices(U * sigma * V.T)
    More here:
    https://en.wikipedia.org/wiki/Singular-value_decomposition
    :param matrix: np.array
    :return: np.array, np.array, np.array
    """
    n, m = matrix.shape

    # Compute eigenvalues of A * A(T)
    AAT = matrix.dot(matrix.T)
    eigenvalues = np.linalg.eigvals(AAT)

    # Compute eigenvectors of A(T) * A
    ATA = matrix.T.dot(matrix)
    values, eigenvectors = np.linalg.eig(ATA)

    # Compute Sigma(S) -> middle diagonal matrix
    S = compute_sigma(eigenvalues, n, m)

    # Compute V -> right orthogonal matrix
    V = compute_V(values, eigenvectors)
    V = V.T

    # Compute U -> left orthogonal matrix
    U = compute_U(matrix, S, V, n)

    return U, S, V.T


def get_A_approximation(U, sigma, V, rank):
    """
    Return an matrix approximation of a specific rank
    based on its SVD decomposition(U, sigma, V)
    :param U:
    :param sigma:
    :param V:
    :param rank:
    :return:
    """

    a = np.matrix(U[:, :rank])
    b = sigma[:rank]
    b = b[:rank, :rank]
    c = np.matrix(V[:rank, :])

    approximation = np.matrix(a * b * c, dtype='float64')

    return approximation


def householder_reduction(A):
    """Perform Golub-Kahan bidiagonalization of matrix A using series
    of orthogonal transformations.
    Args:
        A (numpy.ndarray): 2-dim array representing input matrix of size m by n, where m > n.
    Returns:
        Three 2-dim numpy arrays which correspond to matrices  U, B and V
        such that A = UB(V.T), U and V are orthogonal and B is upper bidiagonal.
    """

    # initialize matrices
    B = np.copy(A)
    m, n = B.shape
    U = np.eye(m)
    V = np.eye(n)
    U_temp = np.eye(m)
    V_temp = np.eye(n)

    for k in range(n):

        # zero out elements under diagonal element in k-th column
        u = np.copy(B[k:m, k])
        u[0] += np.sign(u[0]) * np.linalg.norm(u)
        u = u / np.linalg.norm(u)
        U_temp[k:m, k:m] = np.eye(m - k) - 2 * np.outer(u, u)
        # update matrix U
        U[k:m, :] = np.matmul(U_temp[k:m, k:m], U[k:m, :])
        B[k:m, k:n] = np.matmul(U_temp[k:m, k:m], B[k:m, k:n])

        # zero out elements to the right of right neighbour of diagonal entry in k-th row
        if k <= n - 2:
            v = np.copy(B[k, (k + 1): n])
            v[0] += np.sign(v[0]) * np.linalg.norm(v)
            v = v / np.linalg.norm(v)
            V_temp[k + 1:n, k + 1:n] = np.eye(n - k - 1) - 2 * np.outer(v, v)
            # update matrix V.T
            V[:, k + 1:n] = np.matmul(V[:, k + 1:n], V_temp[k + 1:n, k + 1:n].T)
            B[k:m, (k + 1):n] = np.matmul(B[k:m, (k + 1):n], V_temp[k + 1:n, k + 1: n].T)

    return U.T, B, V


def two_dim_evs(A):
    tr = A[0, 0] + A[1, 1]
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    d = np.sqrt(tr ** 2 - 4 * det)
    return (tr + d) / 2, (tr - d) / 2


def golub_kahan_step(B, U, V, p, q):
    m, n = B.shape
    C = np.zeros((2, 2))
    C[0, 0] = np.dot(B[p:n - q, n - q - 2], B[p:n - q, n - q - 2])
    C[1, 0] = C[0, 1] = np.dot(B[p:n - q, n - q - 2], B[p:n - q, n - q - 1])
    C[1, 1] = np.dot(B[p:n - q, n - q - 1], B[p:n - q, n - q - 1])

    lambda1, lambda2 = two_dim_evs(C)
    mu = lambda1 if (np.abs(lambda1 - C[1, 1]) < np.abs(lambda2 - C[1, 1])) else lambda2
    alpha, beta = B[p, p] ** 2 - mu, B[p, p] * B[p, p + 1]

    R = np.zeros((2, 2))
    for k in range(p, n - q - 1):
        r = np.sqrt(alpha ** 2 + beta ** 2)
        c, s = alpha / r, -beta / r

        R[0, 0] = c
        R[0, 1] = s
        R[1, 0] = -s
        R[1, 1] = c
        B[:, k:k + 2] = np.matmul(B[:, k:k + 2], R)
        V[k:k + 2, :] = np.matmul(R.T, V[k:k + 2, :])

        alpha, beta = B[k, k], B[k + 1, k]
        r = np.sqrt(alpha ** 2 + beta ** 2)
        c, s = alpha / r, -beta / r

        R[0, 0] = c
        R[0, 1] = s
        R[1, 0] = -s
        R[1, 1] = c

        B[k:k + 2, :] = np.matmul(R.T, B[k:k + 2, :])
        U[:, k:k + 2] = np.matmul(U[:, k:k + 2], R)
        if k < n - q - 2:
            alpha, beta = B[k, k + 1], B[k, k + 2]
    return U, B, V


def givens_push(U, B, p, q):
    m, n = B.shape
    for i in range(p, n - q - 1):
        if B[i, i] == 0:
            for j in range(i + 1, n - q):
                alpha, beta = B[i, j], B[j, j]
                r = np.sqrt(alpha ** 2 + beta ** 2)
                c, s = beta / r, -alpha / r

                temp = c * B[i] + s * B[j]
                B[j] = -s * B[i] + c * B[j]
                B[i] = temp

                temp = c * U[:, i] + s * U[:, j]
                U[:, j] = -s * U[:, i] + c * U[:, j]
                U[:, i] = temp
    return U, B


def svdF(A):
    m, n = A.shape
    epsilon = 1e-3
    U, B, V = householder_reduction(A)
    V = V.T
    while True:
        for i in range(n - 1):
            if np.abs(B[i, i + 1]) <= epsilon * (np.abs(B[i, i]) + np.abs(B[i + 1, i + 1])):
                B[i, i + 1] = 0

        q = 0
        while q < n - 1 and B[n - q - 2, n - q - 1] == 0:
            q += 1

        q = q if q != (n - 1) else (q + 1)

        if q == n:
            break

        r = 0
        while r < n - q - 1 and B[n - q - r - 2, n - q - r - 1]:
            r += 1
        r += 1

        p = n - q - r
        if any([B[i, i] for i in range(p, n - q)]):
            U, B, V = golub_kahan_step(B, U, V, p, q)
        else:
            U, B = givens_push(U, B, p, q)

    for i in range(n):
        if B[i, i] < 0:
            U[:, i] *= -1
            B[i, i] *= -1
    return U, B, V



#calc eigen vals & vect by using qr method
def qrmethod(m):
    x = m
    y = np.eye(m.shape[0])
    for i in range(360):
        q, r = np.linalg.qr(x)
        y = np.dot(y, q)
        x = np.dot(r, q)
    return np.diag(x), y

#calc right singular vect(MtM)
def findRightSing(m):
    return np.dot(m.transpose(), m)

#get sigma
def getSigma(n,mat,val):
    for i in range(n):
        mat[i] = np.sqrt(abs(val[i]))
#get U
def getU(u,n,val):
    for i in range (n):
        u[:,i] = u[:,i]/np.sqrt(np.abs(val[i]))

#decompose m -> U sigma Vt
def svdmethod(m):
    #init
    V = findRightSing(m)
    eigVal, eigVec = qrmethod(V)

    #sort eigen vals & vec
    idx = eigVal.argsort()[::-1]
    eigVal = eigVal[idx]
    eigVec = eigVec[:,idx]
    eigVal = eigVal[eigVal != 0.0] #discard 0
    k = len(eigVal)

    #init U, sigma, Vt
    U = np.dot(m,eigVec[:,:k])
    sigma = np.zeros(k)
    Vt = eigVec.transpose()

    #get U, sigma
    getSigma(k,sigma,eigVal)
    getU(U,k,eigVal)
    return U, sigma, Vt


EPS = 1e-10 # Maximum precision of the calculation
TOL = 1e-15 # Maximum precision of the machine
MAX_ITERATION = 100 # Maximum number of iterations in SVD algorithm

def get_sign(x):
    '''returns sign of x'''
    if x >= 0:
        return 1
    return -1

def householder_qr(A):
    '''returns QR factorization of A as (Q, R)'''
    (nrow, ncol) = np.shape(A)
    Q = np.identity(nrow, dtype=np.float64)
    R = np.copy(A)
    for row in range(nrow):
        x = R[row:nrow, row]
        
        # y is the desired reflection of x
        y = np.zeros_like(x, dtype=float)
        y[0] = -get_sign(x[0]) * np.linalg.norm(x)
        
        # u is the householder vector for desired reflection transform
        u = np.zeros(nrow)
        u[row:nrow] = y - x
        u = u / np.linalg.norm(u)
        
        H = np.identity(nrow, dtype=np.float64) - 2*np.outer(u, u)
        Q = Q @ H
        R = H @ R
    return (Q, R)

def householder_bidiagonalization(A):
    '''returns Bidiagnoliziation of A as (U,A,V*)'''
    # For now, I assume ncol <= nrow
    
    (nrow, ncol) = np.shape(A)
    U = np.identity(nrow, dtype=np.float64)
    V = np.identity(ncol, dtype=np.float64)
    R = np.copy(A)
    
    col = 0
    for row in range(nrow):
        x = R[row:nrow, row]
        
        # y is the desired reflection of x
        y = np.zeros_like(x, dtype=float)
        y[0] = -get_sign(x[0]) * np.linalg.norm(x)

        # u is the householder vector for desired reflection transform
        u = np.zeros(nrow)
        u[row:nrow] = y - x
        u = u / np.linalg.norm(u)
        
        H = np.identity(nrow, dtype=np.float64) - 2*np.outer(u, u)
        U = U @ H
        R = H @ R
        
        if row+1 < ncol:
            x = R[row, row+1:ncol]

            # y is the desired reflection of x
            y = np.zeros_like(x, dtype=float)
            y[0] = -get_sign(x[0]) * np.linalg.norm(x)

            # u is the householder vector for desired reflection transform
            u = np.zeros(ncol)
            u[row+1:ncol] = y - x
            u = u / np.linalg.norm(u)

            H = np.identity(nrow, dtype=np.float64) - 2*np.outer(u, u)

            V = H @ V
            R = R @ H
    
    for row in range(nrow):
        for col in range(ncol):
            if abs(R[row, col]) < EPS: R[row, col] = 0.0
                
    return (U, R, V)


def blocking(B):
    ''' B must be a square upper bidiagonal matrix'''
    n = len(B)

    # identify size of larget diagonal sub matrix 
    q = 0
    if abs(B[n-2, n-1]) < EPS:
        q = 1
    while q > 0 and q < n:
        if n-q-2>=0 and abs(B[n-q-2, n-q-1]) > EPS: 
            break
        q = q + 1

    # identify larget submatrix with non-zero superdiagonal entries.
    p = n - q
    while p > 0:
        if p-2>=0 and abs(B[p-2, p-1]) < EPS:
            p = p - 1
            break;
        p = p-1

    B_1 = B[0:p, 0:p]
    B_2 = B[p:n-q, p:n-q]
    B_3 = B[n-q:, n-q:]
    return (B_1, B_2, B_3)


def wilkinson_2by2(C):
    '''Extract an approximate Wilkinson shift for 2 by 2 matrix C'''

    delta = (C[0,0]+C[1,1])**2 - 4*(C[0,0]*C[1,1]-C[0,1]*C[1,0])
    if delta > 0:
        lambda_1 = ((C[0,0]+C[1,1]) + math.sqrt(delta))/2
        lambda_2 = ((C[0,0]+C[1,1]) - math.sqrt(delta))/2
        if abs(lambda_1 - C[1,1]) < abs(lambda_2 - C[1,1]):
            return lambda_1
        return lambda_2
    return C[1,1]


def givens(i, j, _n, x, y):
    ''' 
        Returns a Givens rotation matrix with size n 
        The rotation zero out j-th row for a column c that c[i] = x, c[j] = y
    '''
    G = np.identity(_n, dtype=np.float64)
    norm = math.sqrt(x**2 + y**2)
    c = x / norm
    s = y / norm
    G[i, i], G[i, j], G[j, i], G[j, j] = c, s, -s, c

    return G
    
    
def svd(A):
    ''' For now I assume that A is a square matrix'''
    (nrow, ncol) = np.shape(A)
    (U, B, V) = householder_bidiagonalization(A)
    n = len(B)

    for iteration in range(MAX_ITERATION):
        # Zero out small superdiagonal entries
        for i in range(n-1):
            if abs(B[i, i+1]) < EPS*(abs(B[i, i])+abs(B[i+1, i+1])):
                B[i, i+1] = 0.0
        
        # Zero out small entries
        for i in range(n):
            for j in range(n):
                if abs(B[i, j]) < TOL:
                    B[i, j] = 0.0
      
        # Split B into blocks
        (B_1, B_2, B_3) = blocking(B)
        
        # We have found the decomposition!
        if len(B_3) == n:
            return (U, B, V)
        
        zero_diagonal_entry = False
        for i in range(len(B_2)-1):
            if B_2[i, i] == 0:
                ''' We use Givens rotations to zero out the upperdiagonal entry in i-th row'''
                U_2 = np.identity(len(B_2), dtype=np.float64)
                for j in range(i, len(B_2)-1):
                    G = givens(j+1, j, n, B[i+1, i+1], B_2[i, i+1])
                    B_2 = G @ B_2
                    U_2 = U_2 @ G.T

                U[len(B_1):len(B_1)+len(B_2), len(B_1):len(B_1)+len(B_2)] = \
                                U[len(B_1):len(B_1)+len(B_2), len(B_1):len(B_1)+len(B_2)] @ U_2
                zero_diagonal_entry = True
                break
        
        if not zero_diagonal_entry:
            ''' If we reach here, we are sure that B_2 has no zero diagonal or superdiagonal entry'''
            n_2 = len(B_2)
            
            U_2 = np.identity(n_2, dtype=np.float64)
            V_2 = np.identity(n_2, dtype=np.float64)
            
            mu = wilkinson_2by2(B_2[n_2-2:, n_2-2:].T @ B_2[n_2-2:, n_2-2:])

            y = B_2[0,0]**2 - mu
            z = B_2[0,0]*B_2[0, 1]
            for i in range(0, n_2-1):
                G = givens(i, i+1, n_2, y, z)

                B_2 = B_2 @ G.T
                V_2 = G @ V_2
                
                y, z = B_2[i,i], B_2[i+1, i]
                G = givens(i, i+1, n_2, y, z)
                B_2 = G @ B_2
                U_2 = U_2 @ G.T
                        
                if i < n_2-2:
                    y, z = B_2[i,i+1], B_2[i,i+2]
            
        B[len(B_1):len(B_1)+len(B_2), len(B_1):len(B_1)+len(B_2)] = B_2
        U_2_extended = np.identity(n)
        V_2_extended = np.identity(n)
        U_2_extended[len(B_1):len(B_1)+len(B_2), len(B_1):len(B_1)+len(B_2)] = U_2
        V_2_extended[len(B_1):len(B_1)+len(B_2), len(B_1):len(B_1)+len(B_2)] = V_2
        
        U = U@U_2_extended
        V = V_2_extended@V
        # END OF ITERATION
            
    return (U, B, V)

img = np.asarray(Image.open(r'jupiter.jpg'))
k = int(input("compression percentage: "))

r = img[:,:,0]  # array for R
g = img[:,:,1]  # array for G
b = img[:,:,2] # array for B



print("compressing...")
    
    # Calculating the svd components for all three arrays
#ur,sr,vr = svd(r)
#ug,sg,vg = svd(g)
#ub,sb,vb = svd(b)
# ur,sr,vr = np.linalg.svd(r, full_matrices=False)
# ug,sg,vg = np.linalg.svd(g, full_matrices=False)
# ub,sb,vb = np.linalg.svd(b, full_matrices=False)

# ur,sr,vr = svdNando(r)
# ug,sg,vg = svdNando(g)
# ub,sb,vb = svdNando(b)
q = 1
p = 5

Ur, Sr, VTr = rSVD(r,k,q,p)
Ug, Sg, VTg = rSVD(g,k,q,p)
Ub, Sb, VTb = rSVD(b,k,q,p)
# XrSVD = rU[:,:(r+1)] @ np.diag(rS[:(r+1)]) @ rVT[:(r+1),:]

k = round((k/100) * Sb.shape[0])

    # Forming the compress image with reduced information
    # We are selecting only k singular values for each array to make image which will exclude some information from the 
    # image while image will be of same dimension
    
    # ur (mxk), diag(sr) (kxk) and vr (kxn) if image is off (mxn)
    # so let suppose we only selecting the k1 singular value from diag(sr) to form image
#rr = ur @ sr @ vr
#rg = ug @ sg @ vg
#rb = ub @ sb @ vb
rr = np.dot(Ur[:,:k],np.dot(np.diag(Sr[:k]), VTr[:k,:]))
rg = np.dot(Ug[:,:k],np.dot(np.diag(Sg[:k]), VTg[:k,:]))
rb = np.dot(Ub[:,:k],np.dot(np.diag(Sb[:k]), VTb[:k,:]))
    
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
compressed_image.save("anotherx.jpg")
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
