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
    cdef float[:,:,:] R
    cdef float[:,:,:] Cost
    cdef float[:,:] min_cost

    def __init__(self, P1, P2):
        self.P1 = P1
        self.P2 = P2

    cdef float cost_func(self, int n, float * R, float * Cost, float * PrevCost, float prev_min):
        cdef float cur_min, v
        cdef float P1 = self.P1, P2 = self.P2
        cdef int k
        cur_min = INF
        for k in xrange(1, n-1):
            v = PrevCost[k]
            v = min(v, P1 + PrevCost[k-1])
            v = min(v, P1 + PrevCost[k+1])
            v = min(v, P2 + prev_min)
            v += R[k-1] - prev_min
            Cost[k] = v
            cur_min = min(v, cur_min)
        return cur_min

    def calc_cost(self, float[:,:,:] TotalCost, int di, int dj):
        cdef int h, w, d
        h, w, d = self.R.shape[0], self.R.shape[1], self.R.shape[2]
        self.Cost = np.zeros((2, w+2, d+2), np.float32)
        self.Cost[:,:,0], self.Cost[:,:,d+1] = INF, INF
        self.min_cost = np.zeros((2, w+2), np.float32)
        cdef bint flip = (di < 0) or (di == 0 and dj < 0)
        
        cdef int i, j, i1, j1, k, pi, pj
        for i1 in xrange(h):
            for j1 in xrange(w):
                i, j = i1, j1
                if flip:
                    i, j = h-i1-1, w-j1-1
                pi = i - di
                pj = j - dj

                self.min_cost[i&1, 1+j] = self.cost_func(self.Cost.shape[2], &self.R[i,j,0], 
                    &self.Cost[i&1, 1+j, 0], 
                    &self.Cost[pi&1, 1+pj, 0], 
                    self.min_cost[pi&1, 1+pj])
                for k in xrange(d):
                    TotalCost[i, j, k] += self.Cost[i&1, 1+j, 1+k]

    def __call__(self, R, paths=8):
        paths8  = [(1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1)]
        if paths == 8:
            paths = paths8

        self.R = R
        TotalCost = np.zeros_like(R)
        for di, dj in paths:
            self.calc_cost(TotalCost, di, dj)
        return TotalCost



#Ecdef class GeodSGM(SGM):
#E    def __init__(self, P1, P2):
#E        SGM.__init__(self, P1, P2)
#E
#E    cdef float cost_func(self, int n, float * R, float * Cost, float * PrevCost, float prev_min):
#E        cdef float cur_min, v, C
#E        cdef float P1 = self.P1, P2 = self.P2
#E        cdef int k
#E        cur_min = INF
#E        for k in xrange(1, n-1):
#E            C = R[k-1]
#E            v = calc_t0(PrevCost[k], PrevCost[k-1], C, P1)
#E            v = min(v, calc_t0(PrevCost[k], PrevCost[k+1], C, P1))
#E            v = min(v, P2 + prev_min)
#E            v -= prev_min
#E
#E            Cost[k] = v
#E            cur_min = min(v, cur_min)
#E        return cur_min


#cdef class SGM_P2:
#    cdef float P1
#
#    def __init__(self, P1, img):
#        self.P1 = P1
#        self.P2 = P2
#
#    def calc_p2
#
#    cdef float cost_func(self, int n, float * R, float * Cost, float * PrevCost, float prev_min):
#        cdef float cur_min, v
#        cdef float P1 = self.P1, P2 = self.P2
#        cdef int k
#        cur_min = INF
#        for k in xrange(1, n-1):
#            v = PrevCost[k]
#            v = min(v, P1 + PrevCost[k-1])
#            v = min(v, P1 + PrevCost[k+1])
#            v = min(v, P2 + prev_min)
#            v += R[k-1] - prev_min
#            Cost[k] = v
#            cur_min = min(v, cur_min)
#        return cur_min


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
