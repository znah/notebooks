#cython: boundscheck=False
#cython: wraparound=False

from libc.math cimport sqrt

import numpy as np


ctypedef unsigned char uint8_t
ctypedef unsigned int uint32_t

DEF INF = 1e+9

cdef float calc_t0(float t1, float t2, float C, float P):
    cdef float D = 1.0 - ((t1-t2) / P)**2
    if D < 0.0 or t1 < t2:
        return min(t1 + C, t2 + sqrt(C*C + P*P))
    return t1 + C*sqrt(D)

cdef class SGM:
    cdef float P1
    cdef float P2

    def __init__(self, P1, P2):
        self.P1 = P1
        self.P2 = P2

    cdef float cost_func(self, float[:] R, float[:] Cost, float[:] PrevCost, float prev_min):
        cdef float cur_min, v
        cdef float P1 = self.P1, P2 = self.P2
        cdef int k
        cur_min = INF
        for k in xrange(1, Cost.shape[0]-1):
            v = PrevCost[k]
            v = min(v, P1 + PrevCost[k-1])
            v = min(v, P1 + PrevCost[k+1])
            v = min(v, P2 + prev_min)
            v += R[k-1] - prev_min
            Cost[k] = v
            cur_min = min(v, cur_min)
        return cur_min

    def calc_cost(self, float[:,:,:] R, float[:,:,:] TotalCost, int di, int dj):
        cdef int h, w, d
        h, w, d = R.shape[0], R.shape[1], R.shape[2]
        cdef float[:,:,:] Cost = np.zeros((2, w+2, d+2), np.float32)
        Cost[:,:,0], Cost[:,:,d+1] = INF, INF
        cdef float[:,:] min_cost = np.zeros((2, w+2), np.float32)
        cdef bint flip = (di < 0) or (di == 0 and dj < 0)
        
        cdef int i, j, i1, j1, k
        for i1 in xrange(h):
            for j1 in xrange(w):
                i, j = i1, j1
                if flip:
                    i, j = h-i1-1, w-j1-1
                min_cost[i&1, 1+j] = self.cost_func(R[i,j], Cost[i&1, 1+j], Cost[(i-di)&1, 1+j-dj], min_cost[(i-di)&1, 1+j-dj])
                for k in xrange(d):
                    TotalCost[i, j, k] += Cost[i&1, 1+j, 1+k]

    def __call__(self, R):
        Cost = np.zeros_like(R)
        for di, dj in [(1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1)]:
            self.calc_cost(R, Cost, di, dj)
        return Cost



cdef class GeodSGM(SGM):
    def __init__(self, P1, P2):
        SGM.__init__(self, P1, P1)


    cdef float cost_func(self, float[:] R, float[:] Cost, float[:] PrevCost, float prev_min):
        cdef float cur_min, v, C
        cdef float P1 = self.P1, P2 = self.P2
        cdef int k
        cur_min = INF
        for k in xrange(1, Cost.shape[0]-1):
            C = R[k-1]
            v = calc_t0(PrevCost[k], PrevCost[k-1], C, P1)
            v = min(v, calc_t0(PrevCost[k], PrevCost[k+1], C, P1))
            v = min(v, P2 + prev_min)
            v -= prev_min

            Cost[k] = v
            cur_min = min(v, cur_min)
        return cur_min




def census3(uint8_t[:,:] A):
    cdef uint8_t[:,:] C = np.zeros_like(A)
    cdef int h, w, i, j
    cdef uint8_t c
    h, w = A.shape[0], A.shape[1]
    for i in xrange(1, h-1):
        for j in xrange(1, w-1):
            v = A[i, j]
            c = A[i-1, j-1] > v
            c = (A[i-1, j  ] > v) | (c<<1)
            c = (A[i-1, j+1] > v) | (c<<1)
            c = (A[i  , j-1] > v) | (c<<1)
            c = (A[i  , j+1] > v) | (c<<1)
            c = (A[i+1, j-1] > v) | (c<<1)
            c = (A[i+1, j  ] > v) | (c<<1)
            c = (A[i+1, j+1] > v) | (c<<1)
            C[i, j] = c
    return np.asarray(C)

def census5(uint8_t[:,:] A):
    cdef uint32_t[:,:] C = np.zeros_like(A, np.uint32)
    cdef int h, w, i, j, i1, j1
    cdef uint32_t c
    h, w = A.shape[0], A.shape[1]
    for i in xrange(2, h-2):
        for j in xrange(2, w-2):
            v = A[i, j]
            c = 0
            for i1 in xrange(-2, 3):
                for j1 in xrange(-2, 3):
                    c = (A[i+i1, j+j1] > v) | (c<<1)
            C[i, j] = c
    return np.asarray(C)
