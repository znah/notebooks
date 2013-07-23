#cython: boundscheck=False
#cython: wraparound=False

from libc.math cimport sqrt

import numpy as np

DEF INF = 1e+9

cdef float calc_t0(float t1, float t2, float C, float P):
    cdef float D = 1.0 - ((t1-t2) / P)**2
    if D < 0.0 or t1 < t2:
        return min(t1 + C, t2 + sqrt(C*C + P*P))
    return t1 + C*sqrt(D)

cdef float cost_func(float[:] R, float[:] Cost, float[:] PrevCost, float prev_min, float D1, float D2):
    cdef float cur_min, v, C
    cdef int k
    cur_min = INF
    for k in xrange(1, Cost.shape[0]-1):
        C = R[k-1]
        v = calc_t0(PrevCost[k], PrevCost[k-1], C, D1)
        v = min(v, calc_t0(PrevCost[k], PrevCost[k+1], C, D1))
        v = min(v, D2 + prev_min)
        #v -= prev_min

        #v = PrevCost[k]
        #v = min(v, D1 + PrevCost[k-1])
        #v = min(v, D1 + PrevCost[k+1])
        #v = min(v, D2 + prev_min)
        #v += R[k-1] - prev_min
        
        Cost[k] = v
        cur_min = min(v, cur_min)
    return cur_min

def calc_cost(float[:,:,:] R, float[:,:,:] TotalCost, float D1, float D2, int di, int dj):
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
            min_cost[i&1, 1+j] = cost_func(R[i,j], Cost[i&1, 1+j], Cost[(i-di)&1, 1+j-dj], min_cost[(i-di)&1, 1+j-dj], D1, D2)
            for k in xrange(d):
                TotalCost[i, j, k] += Cost[i&1, 1+j, 1+k]


def SGM(R, D1, D2):
    Cost = np.zeros_like(R)
    for di, dj in [(1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1)]:
        calc_cost(R, Cost, D1, D2, di, dj)
    return Cost

