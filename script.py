import cv2
import sys
from matplotlib import pyplot as plt
from masktect.tracker.find_faces import find_faces


inp = cv2.imread(sys.argv[1])
res = find_faces(inp, [])
plt.imshow(cv2.cvtColor(res[0], cv2.COLOR_BGR2RGB))
plt.show()
