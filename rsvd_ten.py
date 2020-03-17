import numpy as np
import time
import itertools
STR = 'abcdefghijklmnopqrstuvwxyz'

def orthonormal_ten(v,split):
    '''
    Make the "columns" of a tensor orthonormal
    '''
    # Copy matrix
    u = v.copy()
    shape = v.shape

    # Make Orthogonal
    einstr = STR[:split]+','+STR[:split]+'->'
    for k in itertools.product(*[range(i) for i in shape[split:]]):
        for j in itertools.product(*[range(k[i]) for i in range(len(k))]):
            kind = (Ellipsis,)+k
            jind = (Ellipsis,)+j
            u[kind] -= np.einsum(einstr,v[kind],u[jind])/np.einsum(einstr,u[jind],u[jind])*u[jind]

    # Make normal
    einstr = STR[:split]+','+STR[:split]+'->'
    for k in itertools.product(*[range(i) for i in shape[split:]]):
        kind = (Ellipsis,)+k
        u[kind] /= np.sqrt(np.einsum(einstr,u[kind],u[kind]))

    # Return Result
    return u

def get_range_ten(A,split,l,q=2):
    '''
    Compute a range that approximates the range of tensor A
    Reference: Algorithm 4.4 of https://arxiv.org/abs/0909.4061

    args:
        A: nd array
            The matrix whose range we are finding
        split: int
            The index along which the svd will be done
        l: int
            The number of vectors to be used (output will be size m x l)

    kwargs:
        q: int
            The number of iterations to perform (default = 2)

    returns:
        Q: ndarray
            An orthonormal tensor whose range
            approximates the range of the input tensor A
    '''
    # Get tensor dimensions
    shape = A.shape
    ndiml = len(shape[:split])
    ndimr = len(shape[split:])

    # Generate Random Gaussian Matrix
    Omega = np.random.normal(size=A.shape[split:]+(l,))

    # Initial Setup
    einstr = STR[:ndiml+ndimr]+','\
             +STR[ndiml:ndiml+ndimr+1]+'->'\
             +STR[:ndiml]+STR[ndiml+ndimr:ndiml+ndimr+1]
    Y = np.einsum(einstr,A,Omega)
    Q = orthonormal_ten(Y,ndiml)

    # Loop through iterations
    for i in range(q):

        # First step
        einstr =  STR[:ndiml+ndimr]+','\
                 +STR[:ndiml]+STR[ndiml+ndimr:ndiml+ndimr+1]+'->'\
                 +STR[ndiml:ndiml+ndimr+1]
        Y = np.einsum(einstr,A,Q)
        Q = orthonormal_ten(Y,ndimr)

        # Second Step
        einstr = STR[:ndiml+ndimr]+','\
                 +STR[ndiml:ndiml+ndimr+1]+'->'\
                 +STR[:ndiml]+STR[ndiml+ndimr:ndiml+ndimr+1]
        Y = np.einsum(einstr,A,Q)
        Q = orthonormal_ten(Y,ndiml)

    # Return Result
    return Q

def rsvd_ten(A,split,k,q=2,nover=2):
    '''
    Compute the svd of a tensor using the randomized svd algorithm
    detailed in https://arxiv.org/abs/0909.4061. Here, we use algorithm
    4.4 for step A and algorithm 5.1 for step B

    args:
        A: Matrix
            The matrix whose range we are finding
            of size m x n
        split: int
            The index along which the svd will be done
        l: int
            The number of vectors to be used (output will be size m x l)

    kwargs:
        q: int
            The number of iterations to perform (default = 2)
        nover: int
            The number of extra vectors to be used (will be truncated off 
            at the end)

    returns:
        Q: ndarray
            The resulting unitary U tensor from SVD
        R: ndarray
            The resulting S*V tensor from SVD
    '''
    # Do Step A
    Q = get_range_ten(A,split,k+nover,q=q)

    # Do Step B
    shape = A.shape
    ndiml = len(shape[:split])
    ndimr = len(shape[split:])
    einstr =  STR[:ndiml+1] + ','\
             +STR[:ndiml] + STR[ndiml+1:ndiml+ndimr+1] + '->'\
             +STR[ndiml:ndiml+ndimr+1]
    B = np.einsum(einstr,Q,A)
    # Truncate to k
    Q,B = Q[...,:k],B[:k,...]
    return Q,B
