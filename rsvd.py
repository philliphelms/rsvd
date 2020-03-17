import numpy as np
import time

def orthonormal_mat(v):
    # Copy matrix
    u = v.copy()
    m,n = v.shape # m rows, n cols
    # Make Orthogonal
    for k in range(n):
        for j in range(k):
            u[:,k] -= np.einsum('i,i->',v[:,k],u[:,j])/np.einsum('i,i->',u[:,j],u[:,j])*u[:,j]
    # Make normal
    for k in range(n):
        u[:,k] /= np.sqrt(np.einsum('i,i->',u[:,k],u[:,k]))
    return u

def get_range(A,l,q=2):
    '''
    Compute a range that approximates the range of A
    Reference: Algorithm 4.4 of https://arxiv.org/abs/0909.4061

    args:
        A: Matrix
            The matrix whose range we are finding
            of size m x n
        l: int
            The number of vectors to be used (output will be size m x l)

    kwargs:
        q: int
            The number of iterations to perform (default = 2)

    returns:
        Q: Matrix
            A (m x l) sized orthonormal matrix whose range
            approximates the range of the input matrix A
    '''
    # Get matrix dimensions
    m,n = A.shape
    # Generate Random Gaussian Matrix
    Omega = np.random.normal(size=(n,l))
    # Initial Setup
    Y = np.einsum('ij,jk->ik',A,Omega)
    #Q,R = np.linalg.qr(Y)
    Q = orthonormal_mat(Y)
    # Loop through iterations
    for i in range(q):
        # First step
        Y = np.einsum('ij,jk->ik',A.T,Q)
        #Q,R = np.linalg.qr(Y)
        Q = orthonormal_mat(Y)
        # Second Step
        Y = np.einsum('ij,jk->ik',A,Q)
        #Q,R = np.linalg.qr(Y)
        Q = orthonormal_mat(Y)
    return Q

def rsvd(A,k,q=2,nover=2):
    '''
    Compute the svd of a matrix using the randomized svd algorithm
    detailed in https://arxiv.org/abs/0909.4061. Here, we use algorithm
    4.4 for step A and algorithm 5.1 for step B

    args:
        A: Matrix
            The matrix whose range we are finding
            of size m x n
        l: int
            The number of vectors to be used (output will be size m x l)

    kwargs:
        q: int
            The number of iterations to perform (default = 2)

    returns:
        U: Matrix
            The resulting unitary U matrix from SVD
        S: Vector
            The Singular value vector
        V: Matrix
            The resulting unitary V matrix from SVD
    '''
    # Do Step A
    t0 = time.time()
    Q = get_range(A,k+nover,q=q)
    t1 = time.time()
    print('Get Range time {}'.format(t1-t0))
    # Do Step B
    t0 = time.time()
    B = np.einsum('ij,jk->ik',Q.T,A)
    U,S,V = np.linalg.svd(B,full_matrices=False)
    U = np.einsum('ij,jk->ik',Q,U)
    t1 = time.time()
    print('Step B time {}'.format(t1-t0))
    # Truncate to k
    U,S,V = U[:,:k],S[:k],V[:k,:]
    return U,S,V
