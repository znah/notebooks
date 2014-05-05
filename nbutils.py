
import cv2
from IPython.display import display, Image

def showarray(a):
    _, data = cv2.imencode('.png', a)
    display(Image(data.tostring(), format='png'))