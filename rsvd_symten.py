from symtensor.settings import load_lib
from symtensor.sym import einsum,normal
from symtensor.misc import SIGN,DEBUG,FLIP
from numpy import argsort,where,array,prod,sqrt
import time
import itertools
STR = 'abcdefghijklmnopqrstuvwxyz'

def orthonormal_ten(v,split):
    '''
    Make the "columns" of a symtensor orthonormal
    '''
    # Load lib for einsum
    lib = load_lib(v.backend)

    # Collect the array
    u = v.copy()
    _v = v.array
    _u = _v.copy()
    
    # Figure out indices we need to iterate over
    iterinds = [range(v.array.shape[i]) for i in range(split,v.ndim-1)] + [range(v.array.shape[u.ndim-1+i]) for i in range(split,v.ndim)]

    # Do the orthogonalization
    for k in itertools.product(*iterinds):

        # Figure out k indices
        kind = tuple()
        for i in range(split):
            kind += (slice(0,v.array.shape[i]),)
        kind += tuple([k[i] for i in range(v.ndim-1-split)])
        for i in range(split):
            kind += (slice(0,v.array.shape[i+v.ndim-1]),)
        kind += tuple([k[i+(v.ndim-1-split)] for i in range(v.ndim-split)])

        for j in itertools.product(*[range(k[i]) for i in range(len(k))]):

            # Figure out j indices
            jind = tuple()
            for i in range(split):
                jind += (slice(0,v.array.shape[i]),)
            jind += tuple([j[i] for i in range(v.ndim-1-split)])
            for i in range(split):
                jind += (slice(0,v.array.shape[i+v.ndim-1]),)
            jind += tuple([j[i+(v.ndim-1-split)] for i in range(v.ndim-split)])

            # Make orthogonal
            vk = _v[kind]
            uj = _u[jind]
            einstr = STR[:uj.ndim]+','+STR[:uj.ndim]+'->'
            _u[kind] -= lib.einsum(einstr,vk,uj)/lib.einsum(einstr,uj,uj)*uj

    # Normalize orthogonal vectors
    for k in itertools.product(*iterinds):
        # Figure out k indices
        kind = tuple()
        for i in range(split):
            kind += (slice(0,v.array.shape[i]),)
        #kind += tuple([k[i] for i in range(split,v.ndim-1)])
        kind += tuple([k[i] for i in range(v.ndim-1-split)])
        for i in range(split):
            kind += (slice(0,v.array.shape[i+v.ndim-1]),)
        kind += tuple([k[i+(v.ndim-1-split)] for i in range(v.ndim-split)])
        _u[kind] /= sqrt(lib.einsum(einstr,_u[kind],_u[kind]))

    # Put back into symtensor
    u.array = _u

    # Return result
    return u

def orthonormal_ten_slow(v,split):
    '''
    Make the "columns" of a symtensor orthonormal
    Not currently used because of extra cost of tranpose
    with ctf tensors. 
    '''
    # Load lib for einsum
    lib = load_lib(v.backend)

    # Collect the array
    u = v.copy()
    _v = v.array
    newinds = []
    for i in range(v.ndim):
        if i < v.ndim-1: newinds.append(i)
        newinds.append(i+v.ndim-1)
    _v = _v.transpose(newinds)
    _u = _v.copy()
    shape = _v.shape
    
    # Make Orthogonal
    split *= 2
    einstr = STR[:split]+','+STR[:split]+'->'
    for k in itertools.product(*[range(i) for i in shape[split:]]):
        for j in itertools.product(*[range(k[i]) for i in range(len(k))]):
            kind = (Ellipsis,)+k
            jind = (Ellipsis,)+j
            _u[kind] -= lib.einsum(einstr,_v[kind],_u[jind])/lib.einsum(einstr,_u[jind],_u[jind])*_u[jind]

    # Make normal
    einstr = STR[:split]+','+STR[:split]+'->'
    for k in itertools.product(*[range(i) for i in shape[split:]]):
        kind = (Ellipsis,)+k
        _u[kind] /= sqrt(lib.einsum(einstr,_u[kind],_u[kind]))
    
    # Put back into symtensor
    newinds = list(range(0,(v.ndim-1)*2,2))+list(range(1,(v.ndim-1)*2,2))+[(v.ndim-1)*2]
    _u = _u.transpose(newinds)
    u.array = _u

    # Return result
    return u

def fused_bonds_qns(sym_range,sign_str,modulus,rhs=0):
    ''' 
    Calculate resulting quantum numbers for fused bonds 
    '''
    qns = []
    all_qns = []
    all_qns_ind = []
    for idx in itertools.product(*sym_range):
        qn = 0
        for ki,i in enumerate(idx):
            qn += SIGN[sign_str[ki]] * sym_range[ki][i]
        qn -= rhs
        if modulus is not None: qn = qn%modulus
        all_qns.append(qn)
        if not (qn in qns): qns.append(qn)
        all_qns_ind += [idx]
    # Sort the results
    inds = list(argsort(all_qns))
    all_qns_ind = [all_qns_ind[i] for i in inds]
    all_qns = [all_qns[i] for i in inds]
    qns.sort()
    # Count Number of each qn
    nqns = [None]*len(qns)
    for i in range(len(qns)):
        nqns[i] = len(where(array(all_qns) == qns[i])[0])
    return all_qns_ind,all_qns,qns,nqns

def make_omega(A,split,l):
    '''
    Make the Omega matrix as a symtensor
    '''
    # Figure out tensor info --------------------------------
    sign_str = ''.join(A.sym[0])
    sym_range = A.sym[1]
    rhs = A.sym[2] # NOTE - Not used currently...
    if rhs is None: rhs = 0
    if not (rhs == 0):
        warn('(rhs != 0) not tested')
    modulus = A.sym[3]
    degen = A.shape
    lib = load_lib(A.backend)
    n_legs = len(degen)

    # Get in and out info 
    split = [list(range(split)),list(range(split,n_legs))]
    in_sym_range = [sym_range[i] for i in split[0]]
    out_sym_range = [sym_range[i] for i in split[1]]
    in_sign_str = ''.join([sign_str[i] for i in split[0]])
    out_sign_str = ''.join([FLIP[sign_str[i]] for i in split[1]])
    in_degen = [degen[i] for i in split[0]]
    out_degen = [degen[i] for i in split[1]]

    # Convert into a block diagonal matrix ------------------
    # Fuse input bonds
    qns_in_all,qns_in_comb,qns_in,nqns_in = fused_bonds_qns(in_sym_range,in_sign_str,modulus,rhs=rhs)
    in_degen_comb = prod(in_degen)
    # Fuse output bonds
    qns_out_all,qns_out_comb,qns_out,nqns_out = fused_bonds_qns(out_sym_range,out_sign_str,modulus)
    out_degen_comb = prod(out_degen)
    # Figure out center bond
    qns_ctr = [value for value in qns_in if value in qns_out]
    ctr_sz  = min(max(nqns_in)*in_degen_comb,max(nqns_out)*out_degen_comb,int(l/len(qns_ctr)))
    if ctr_sz == 0: ctr_sz = 1

    # Create a random (normally distributed) symtensor ------
    Omega = normal(out_degen+[ctr_sz],
                   sym=[out_sign_str+'-',
                        out_sym_range+[qns_ctr],
                        rhs,
                        modulus],
                   backend=A.backend)

    # Return result
    return Omega

def get_range_symten(A,split,l,q=2):
    '''
    Compute a range that approximates the range of tensor A
    Reference: Algorithm 4.4 of https://arxiv.org/abs/0909.4061

    args:
        A: symtensor
            The symtensor whose range we are finding
        split: int
            The index along which the svd will be done
        l: int
            The number of vectors to be used (output will be size m x l)

    kwargs:
        q: int
            The number of iterations to perform (default = 2)

    returns:
        Q: symtensor
            An orthonormal tensor whose range
            approximates the range of the input tensor A
    '''
    # Get tensor dimensions
    ndiml = split
    ndimr = A.ndim-split

    # Generate Random Gaussian Matrix
    Omega = make_omega(A,split,l)

    # Initial Setup
    einstr = STR[:ndiml+ndimr]+','\
             +STR[ndiml:ndiml+ndimr+1]+'->'\
             +STR[:ndiml]+STR[ndiml+ndimr:ndiml+ndimr+1]
    Y = einsum(einstr,A,Omega)
    Q = orthonormal_ten(Y,ndiml)
    #Q = orthonormal_ten_slow(Y,ndiml)

    # Loop through iterations
    for i in range(q):

        # First step
        einstr =  STR[:ndiml+ndimr]+','\
                 +STR[:ndiml]+STR[ndiml+ndimr:ndiml+ndimr+1]+'->'\
                 +STR[ndiml:ndiml+ndimr+1]
        Y = einsum(einstr,A,Q)
        Q = orthonormal_ten(Y,ndimr)

        # Second Step
        einstr = STR[:ndiml+ndimr]+','\
                 +STR[ndiml:ndiml+ndimr+1]+'->'\
                 +STR[:ndiml]+STR[ndiml+ndimr:ndiml+ndimr+1]
        Y = einsum(einstr,A,Q)
        Q = orthonormal_ten(Y,ndiml)

    # Return Result
    return Q

def truncate(ten,ind,sz):
    '''
    Truncate the size of a given tensor leg
    '''
    # Load lib for einsum
    lib = load_lib(ten.backend)

    # Figure out the required quantum sector size
    sz_prev = ten.shape[ind]
    sz_new  = int(sz/len(ten.sym[1][ind]))

    # Create an identity to shrink bond dim
    I = lib.zeros((sz_prev,sz_new),dtype=ten.dtype)
    for i in range(sz_new):
        I[i,i] = 1.

    # Contract tensor with Identity
    einstr =  STR[:len(ten.array.shape)]+','\
             +STR[ten.ndim-1+ind]+STR[ten.ndim-1+ind].upper()+'->'\
             +STR[:ten.ndim-1+ind]+STR[ten.ndim-1+ind].upper()+STR[ten.ndim+ind:len(ten.array.shape)]
    _ten = lib.einsum(einstr,ten.array,I)
    
    # Put back into symtensor
    ten.array = _ten
    ten.shape = _ten.shape[ten.ndim-1:]

    # Return result
    return ten

def rsvd_symten(A,split,k,q=2,nover=2):
    '''
    Compute the svd of a tensor using the randomized svd algorithm
    detailed in https://arxiv.org/abs/0909.4061. Here, we use algorithm
    4.4 for step A and algorithm 5.1 for step B

    args:
        A: symtensor
            The matrix whose range we are finding
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
        Q: symtensor
            The resulting unitary U tensor from SVD
        R: symtensor
            The resulting S*V tensor from SVD
    '''
    # Do Step A
    Q = get_range_symten(A,split,k+nover,q=q)

    # Do Step B
    ndiml = split
    ndimr = A.ndim-split
    einstr =  STR[:ndiml+1] + ','\
             +STR[:ndiml] + STR[ndiml+1:ndiml+ndimr+1] + '->'\
             +STR[ndiml:ndiml+ndimr+1]
    B = einsum(einstr,Q,A)

    # Truncate from l to  k
    Q = truncate(Q,Q.ndim-1,k)
    B = truncate(B,0,k)

    # Return Result
    return Q,B
