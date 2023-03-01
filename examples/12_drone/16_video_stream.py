# -*-coding:utf-8-*-
# Copyright (c) 2020 DJI.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License in the file LICENSE.txt or at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
from robomaster import robot
#from shape_recognize import triangle_recognize

def triangle_recognize(img):

   # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply threshold to the grayscale image
    _, thresh = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Loop through each contour and determine whether it is a triangle
    triangles = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 50 and cv2.contourArea(cnt) < 1000: # set a minimum and maximum area threshold
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) == 3:
                triangles.append(approx)

    img_copy = img.copy()
    cv2.drawContours(img_copy, triangles, -1, (0, 0, 0), 2)
    # cv2.imshow('Identified triangles', img_copy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img_copy

if __name__ == '__main__':
    tl_drone = robot.Drone()
    tl_drone.initialize()

    tl_camera = tl_drone.camera
    # 显示302帧图传
    tl_camera.start_video_stream(display=False)
    tl_camera.set_fps("high")
    tl_camera.set_resolution("high")
    tl_camera.set_bitrate(6)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('video_save.mp4',fourcc,30,(960,720), True)

    for i in range(0, 502):
        img = tl_camera.read_cv2_image()
        # img_copy = triangle_recognize(img)
        #print(img.shape)
        out.write(img)
        cv2.imshow("Drone", img)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    tl_camera.stop_video_stream()

    tl_drone.close()
