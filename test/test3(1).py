# april tag detection
# import apriltag
import pupil_apriltags as apriltag     # for windows
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

img =cv2.imread("./img2.jpg")
# shape = img.sh
img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 创建一个apriltag检测器
# at_detector = apriltag.Detector(apriltag.DetectorOptions(families='tag36h11 tag25h9') )
at_detector = apriltag.Detector(
   families="tag36h11",
   nthreads=1,
   quad_decimate=1.0,
   quad_sigma=0.0,
   refine_edges=1,
   decode_sharpening=0.25,
   debug=0
)
# at_detector = apriltag.Detector(families='tag36h11 tag25h9')  #for windows
# 进行apriltag检测，得到检测到的apriltag的列表
tags = at_detector.detect(gray)
print(type(tags))
print(tags)
print("%d apriltags have been detected."%len(tags))
for tag in tags:
    cv2.circle(img, tuple(tag.corners[0].astype(int)), 4,(255,0,0), 2) # left-top
    cv2.circle(img, tuple(tag.corners[1].astype(int)), 4,(255,0,0), 2) # right-top
    cv2.circle(img, tuple(tag.corners[2].astype(int)), 4,(255,0,0), 2) # right-bottom
    cv2.circle(img, tuple(tag.corners[3].astype(int)), 4,(255,0,0), 2) # left-bottom

cv2.imshow("apriltag_test",img)
cv2.waitKey()



# corners = tags[0].corners


# fig, ax = plt.subplots()
# # ax.imshow(gray, cmap='gray')
# ax.plot(corners[:, 0], corners[:, 1], color='red', linewidth=2)
# ax.add_patch(patches.Polygon(corners, fill=False, edgecolor='green', linewidth=2))
# plt.show()
