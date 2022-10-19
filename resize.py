import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("./submission/lr.png")

resize = cv2.resize(img, (2048,2048))
cv2.imwrite('./submission/test2.png',resize)

print(img)