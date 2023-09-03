import matplotlib.pyplot as plt
import numpy as np
import cv2

with open('config/anchors.txt','r') as f:
    wh = [l.strip().split(',') for l in f.readlines()]

canvas = np.zeros((100,100))

for w,h in wh:
    w, h = float(w), float(h)
    x1, y1, x2, y2 = 100*(0.5-w/2), 100*(0.5-h/2), 100*(0.5+w/2), 100*(0.5+h/2)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(canvas,(x1,y1),(x2,y2),(255,255,255),1)

plt.imshow(canvas)
plt.show()