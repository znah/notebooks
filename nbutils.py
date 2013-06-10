import numpy as np
from IPython.display import Image, display
import PIL.Image

def showarray(a, fn='_tmp.png'):
    if a.dtype in [np.float32, np.float64]:
        a = np.clip(a, 0, 1)*255
    PIL.Image.fromarray(np.uint8(a)).save(fn)
    display( Image(filename=fn) )