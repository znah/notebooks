import numpy as np


def fill(a, i=0, cache=None):
    if cache is None:
        cache = {}
    if a.shape in cache:
        a[:] = cache[a.shape] + i
        return

    if a.shape == (1, 1):
        a[0,0] = i
    else:
        h, w = a.shape
        h2, w2 = h//2, w//2
        n = h2*w2
        fill(np.rot90(a[:h2, :w2])[::-1], i, cache)
        fill(a[:h2, w2:], i+n, cache)
        fill(a[h2:, w2:], i+2*n, cache)
        fill(np.rot90(a[h2:, :w2], -1)[::-1], i+3*n, cache)
    if i == 0:
        cache[a.shape] = a

      

if __name__ == '__main__':
    import pylab as pl
    from time import clock
    n = 1024
    a = np.zeros((n, n), np.int32)
    
    t = clock()
    fill(a)
    print clock()-t


    #pl.imshow(a) 
    #pl.show()

    #y, x = np.indices(a.shape)
    #a = a.ravel()
    #x1 = np.zeros_like(a)
    #y1 = np.zeros_like(a)
    #x1[a] = x.ravel()
    #y1[a] = y.ravel()

    #pl.plot(x1, y1)
    #pl.axis('equal')
    #pl.show()