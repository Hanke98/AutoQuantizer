import numpy as np

def LagrangianCR(grads, r = 0.5, a = None, debug = False):
    '''
    Lagrangian solver with compression rate constraint

    USUAGE: 
    args: 
        grads: g_i * R_i ** 2 (g_i, R_i: square sum of gradients, range; *: element-wise product; index for distinct custom types)
        r: target compression rate in (0,1] compared to float32, i.e., \epsilon_mem
        a: array, particle count for each type. default to (1,..,1)_H of the same dimension of `grads`; 

    returns: tuple(bits, fun)
        bits: resultant fraction bits is in `numpy.array` with int32 type
        fun: error estimation

    '''
    n = len(grads)
    grads = np.array(grads)
    r = r * 32.0 - 1 # 1 for the sign bit
    if a is not None:
        assert(len(a) == n)
        a = np.array(a)
    else:
        a = np.ones(n, dtype = np.float64)


    b = a / grads

    t = np.log2(b)
    
    log_la = -r * 2 - t @ a/np.sum(a) + np.log2(np.log(2))

    la = np.power(2, log_la)
    
    x = b * la / np.log(2)

    fun = x @ grads / 12.0

    bits = -(t + log_la - np.log2(np.log(2)))/2.0
    if debug:
        print((bits+1).sum()/32/a.sum())
    bits = np.floor(bits).astype(np.int32)

    # print(t, b, log_la, -la, x)

    return bits, fun


def LagrangianEB(grads, ref, tol_rel = 1e-3, a = None):
    '''
    Lagrangian solver with error bound constraint

    USUAGE: 
    args: 
        grads: g_i * R_i ** 2 (g_i, R_i: square sum of gradients, range; *: element-wise product; index for distinct custom types)
        ref: reference value, i.e., z
        tol_rel: relative tolerance, i.e., \epsilon_sim
        a: array, particle count for each type. default to (1,..,1)_H of the same dimension of `grads`; 
            
    returns: tuple(bits, fun)
        bits: resultant fraction bits is in `numpy.array` with int32 type
        fun: error bound

    '''
    n = len(grads)
    grads = np.array(grads)
    tol = (ref * tol_rel) ** 2 
    if a is not None:
        assert(len(a) == n)
        a = np.array(a)
    else:
        a = np.ones(n, dtype = np.float64)


    b = a / grads

    x = 12 / np.sum(a) * tol * b                    # x = (\triangle / R)^2

    bits = np.ceil(-np.log2(x)/2.0).astype(np.int32)

    rel = np.sqrt(x @ grads / (12 * ref ** 2))      # relative error bound

    return bits, rel