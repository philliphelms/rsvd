import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sys import argv
from rsvd import *
from rsvd_ten import *
import time

# Set up -----------------------------------------
# Parameters 
fname = argv[1]
k = int(argv[2])
# Load a jpg image
A = np.array(mpl.image.imread(fname))
f = plt.figure()
plt.imshow(A)

# Using matrix np svd -----------------------------------
t0 = time.time()
A1 = np.zeros(A.shape,dtype=A.dtype)
for i in range(3):
    U,S,V = np.linalg.svd(A[:,:,i])
    U,S,V = U[:,:k],S[:k],V[:k,:]
    A1[:,:,i] = np.einsum('ij,j,jk->ik',U,S,V)
t1 = time.time()
print('np matrix svd time {}'.format(t1-t0))
f = plt.figure()
plt.imshow(A1)

# Using matrix randomized svd ---------------------------
t0 = time.time()
A2 = np.zeros(A.shape,dtype=A.dtype)
for i in range(3):
    U,S,V = rsvd(A[:,:,i],k,q=1,nover=10)
    A2[:,:,i] = np.einsum('ij,j,jk->ik',U,S,V)
t1 = time.time()
print('matrix rsvd time {}'.format(t1-t0))
f = plt.figure()
plt.imshow(A2)

# Using reshaped np svd -----------------------------------
t0 = time.time()
(n1,n2,n3) = A.shape
Ar = np.reshape(A,(n1,n2*n3))
U,S,V = np.linalg.svd(Ar,full_matrices=False)
U,S,V = U[:,:k],S[:k],V[:k,:]
Ar = np.einsum('ij,j,jk->ik',U,S,V)
A3 = np.reshape(Ar,(n1,n2,n3))
A3 = A3.astype(A.dtype)
t1 = time.time()
print('np reshaped svd time {}'.format(t1-t0))
f = plt.figure()
plt.imshow(A3)

# Using reshaped rsvd -----------------------------------
t0 = time.time()
(n1,n2,n3) = A.shape
Ar = np.reshape(A,(n1,n2*n3))
U,S,V = rsvd(Ar,k,q=1,nover=10)
Ar = np.einsum('ij,j,jk->ik',U,S,V)
A4 = np.reshape(Ar,(n1,n2,n3))
A4 = A4.astype(A.dtype)
t1 = time.time()
print('reshaped rsvd time {}'.format(t1-t0))
f = plt.figure()
plt.imshow(A4)

# Using Tensor RSVD --------------------------------------
t0 = time.time()
Q,R = rsvd_ten(A,1,k,q=1,nover=k)
A5 = np.einsum('ij,jkl->ikl',Q,R)
A5 = A5.astype(A.dtype)
t1 = time.time()
print('rsvd_ten time {}'.format(t1-t0))
f = plt.figure()
plt.imshow(A5)

plt.show()
