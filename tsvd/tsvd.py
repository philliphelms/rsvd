import numpy as np
import itertools

def gen_split(shape,left=True):
    """
    Generate a splitting tensor 
    """
    # Create the splitting tensor
    if left:
        split_shape = (np.prod(shape),)+shape
    else:
        split_shape = shape+(np.prod(shape),)
    split_ten = np.zeros(split_shape)
    # Turn into correct identity
    allinds = [range(i) for i in shape]
    cnt = 0
    for ind in itertools.product(*allinds):
        if left:
            tenind = (cnt,)+tuple(ind)
        else:
            tenind = tuple(ind)+(cnt,)
        split_ten[tenind] = 1.
        cnt += 1
    # Return result
    return split_ten

# Actual QR functions
def qr_identities(ten,split):
    """
    Do Tensor QR by introducing splitting tensors
    """
    # Get splitting tensors
    shape = ten.shape
    nind = len(shape)
    t0 = time.time()
    split_left = gen_split(shape[:split],left=True)
    split_right = gen_split(shape[split:],left=False)
    tf = time.time()
    print('\tGenerate split tensors time = {}'.format(tf-t0))
    # Combine tensor indices
    letters = 'abcdefghijklmnopqrstuvwxyz'
    t0 = time.time()
    einstr = letters[:split+1] + ',' + \
             letters[1:nind+1] + '->' + \
             letters[0]+letters[split+1:nind+1]
    ten = np.einsum(einstr,split_left,ten)
    einstr = letters[0]+letters[split+1:nind+1] + ',' + \
             letters[split+1:nind+2] + '->' + \
             letters[0] + letters[nind+1]
    ten = np.einsum(einstr,ten,split_right)
    tf = time.time()
    print('\tTensor contraction to mat time {}'.format(tf-t0))
    # Do QR of reshaped tensor
    t0 = time.time()
    Q,R = np.linalg.qr(ten)
    tf = time.time()
    print('\tActual QR time = {}'.format(tf-t0))
    # Split resulting tensor indices
    t0 = time.time()
    einstr = letters[:split+1] + ',' + \
             letters[0] + letters[-1] + '->' + \
             letters[1:split+1] + letters[-1]
    Q = np.einsum(einstr,split_left,Q)
    einstr = letters[-1] + letters[nind+1] + ',' + \
            letters[split+1:nind+2] + '->' + \
            letters[-1] + letters[split+1:nind+1]
    R = np.einsum(einstr,R,split_right)
    tf = time.time()
    print('\tMat contraction to tensor time {}'.format(tf-t0))
    # Return Result
    return Q,R

def qr_reshape(ten,split):
    """
    Normal approach to tensor qr, 
    i.e. reshaping into a matrix then back into tensors
    """
    shape = ten.shape
    matshape = (np.prod(shape[:split]),np.prod(shape[split:]))
    t0 = time.time()
    mat = np.reshape(ten,matshape)
    tf = time.time()
    print('\tInitial Reshape time = {}'.format(tf-t0))
    t0 = time.time()
    Q,R = np.linalg.qr(mat)
    tf = time.time()
    print('\tActual QR time = {}'.format(tf-t0))
    t0 = time.time()
    qshape = shape[:split]+(-1,)
    Q = np.reshape(Q,qshape)
    rshape = (-1,)+shape[split:]
    R = np.reshape(R,rshape)
    tf = time.time()
    print('\tFinal Reshape time = {}'.format(tf-t0))
    return Q,R

if __name__ == "__main__":
    import time
    # Create a random tensor
    chi = 100
    D = 8
    d = 2
    shape = (chi,D,d,D,chi,D)
    split = 3
    ten = np.random.random(shape)
    # ---------------------------------------
    # Do QR with reshape
    print('-'*50)
    t0 = time.time()
    Q,R = qr_reshape(ten,split)
    tf = time.time()
    print('Reshape time {}'.format(tf-t0))
    # ---------------------------------------
    # Do QR with identities
    print('-'*50)
    t0 = time.time()
    Q,R = qr_identities(ten,split)
    tf = time.time()
    print('Identities time {}'.format(tf-t0))
