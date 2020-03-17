import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sys import argv
from rsvd import *
from rsvd_ten import *
import time

# Set up -----------------------------------------
# Parameters 
D = 12
chi = 110
k = int(argv[2])
# Make a structured initial tensor
fname = argv[1]
A = np.array(mpl.image.imread(fname))
Aog = A.copy()
Aog2 = A.copy()
ogshape = A.shape
A = np.reshape(A,(-1))
A = A[:chi*D*D*chi*D]
A = np.reshape(A,(chi,D,D,chi,D))

# Using reshaped np svd -----------------------------------
Ar = np.reshape(A,(chi*D*D,chi*D))
U,S,V = np.linalg.svd(Ar,full_matrices=False)
U,S,V = U[:,:k],S[:k],V[:k,:]
Ar = np.einsum('ij,j,jk->ik',U,S,V)
A1 = np.reshape(Ar,(chi,D,D,chi,D))
A1 = A1.astype(A.dtype)
Aog = np.reshape(Aog,(-1))
Aog[:chi*D*D*chi*D] = A1.reshape((-1))[:chi*D*D*chi*D]
Aog = Aog.reshape(ogshape)
f = plt.figure()
plt.imshow(Aog)

# Using Tensor RSVD --------------------------------------
Q,R = rsvd_ten(A,3,k,q=1,nover=k)
A2 = np.einsum('ijkl,lmn->ijkmn',Q,R)
A2 = A2.astype(A.dtype)
Aog2 = np.reshape(Aog2,(-1))
Aog2[:chi*D*D*chi*D] = A2.reshape((-1))[:chi*D*D*chi*D]
Aog2 = Aog2.reshape(ogshape)
f = plt.figure()
plt.imshow(Aog2)

plt.show()
