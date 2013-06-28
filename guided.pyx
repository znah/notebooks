#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
import cv2

cdef float det3(float *M):
    cdef float M00, M01, M02, M10, M11, M12, M20, M21, M22
    M00, M01, M02 = M[0], M[1], M[2]
    M10, M11, M12 = M[3], M[4], M[5]
    M20, M21, M22 = M[6], M[7], M[8]
    return M00*(M11*M22-M21*M12) - M10*(M01*M22-M21*M02) + M20*(M01*M12-M11*M02)
    
cdef solve3(float *A, float *b, float *x):
    cdef float iD = 1.0 / det3(A)
    cdef float M[3*3]
    M[0:9] = [ b[0], A[1], A[2], b[1], A[4], A[5], b[2], A[7], A[8] ]
    x[0] = det3(M)*iD;
    M[0:9] = [A[0], b[0], A[2], A[3], b[1], A[5], A[6], b[2], A[8]]
    x[1] = det3(M)*iD;
    M[0:9] = [A[0], A[1], b[0], A[3], A[4], b[1], A[6], A[7], b[2]]
    x[2] = det3(M)*iD;
    
  
def calc_a(float[:,:,::1] var_I, float[:,:,:] cov_Ip, float eps):
    cdef float M[3*3]
    cdef float cov[3]
    cdef float a[3]
    cdef int x, y, i
    cdef float * v
    h, w = var_I.shape[0], var_I.shape[1]
    cdef float[:,:,:] A = np.zeros((h, w, 3), np.float32)
    for y in xrange(var_I.shape[0]):
        for x in xrange(var_I.shape[1]):
            v = &var_I[y, x, 0]
            M[0:3] = [v[0]+eps, v[1], v[2]]
            M[3:6] = [v[1], v[3]+eps, v[4]]
            M[6:9] = [v[2], v[4], v[5]+eps]
            for i in xrange(3):
                cov[i] = cov_Ip[y, x, i]
            solve3(M, cov, a)
            for i in xrange(3):
                A[y, x, i] = a[i]
    return np.asarray(A)

def box_r(a, r):
    r = 2*r+1
    return cv2.boxFilter(a, cv2.CV_32F, (r, r), normalize=True, borderType=cv2.BORDER_REFLECT_101)

def guided_filter(I, p, r, eps):
    def box(a):
        return box_r(a, r)
    
    mean_I = box(I)
    mean_p = box(p)
    mean_Ip = box(I*p)
    cov_Ip = mean_Ip - mean_I * mean_p
    
    mean_II = box(I*I)
    var_I = mean_II - mean_I * mean_I
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = box(a)
    mean_b = box(b)
    
    q = mean_a * I + mean_b
    return q

def guided_filter_rgb(I, p, r, eps):
    assert len(I.shape) == 3 and I.shape[-1] == 3
    def box(a):
        return box_r(a, r)

    mean_I = box(I)
    mean_p = box(p)
    mean_Ip = box(I*p[...,np.newaxis])
    cov_Ip = mean_Ip - mean_I * mean_p[...,np.newaxis]

    h, w, chn = I.shape
    var_I = np.zeros((h, w, 6), np.float32)
    for c, (i, j) in enumerate([(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]):
        var_I[...,c] = box(I[...,i]*I[...,j]) - mean_I[...,i]*mean_I[...,j]
        
    a = calc_a(var_I, cov_Ip, eps)
    b = mean_p - (mean_I * a).sum(-1)

    mean_a = box(a)
    mean_b = box(b)
    
    q = (mean_a * I).sum(-1) + mean_b
      
    return q
           
    