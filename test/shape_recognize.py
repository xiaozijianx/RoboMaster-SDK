import cv2
import numpy as np
import sys
import pupil_apriltags as apriltag     # for windows
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def circle_recognize(img):
    #circle
    # Convert the image to HSV color space
    #lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    #bgr = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    # Define the lower and upper bounds of the green color
    green_lower = np.array([45, 100, 100])
    green_upper = np.array([75, 255, 255])

    # Define the lower and upper bounds of the red color
    red_lower = np.array([0, 100, 100])
    red_upper = np.array([10, 255, 255])
    
    
    # Define the lower and upper bounds of the yellow color in BGR
    # lower_yellow = np.array([0, 0, 200])
    # upper_yellow = np.array([50, 50, 255])
    
    lower_yellow = np.array([0, 200, 200])
    upper_yellow = np.array([30, 255, 255])
    
    # Threshold the image to extract the green and red areas
    # green_mask = cv2.inRange(lab, green_lower, green_upper)
    # red_mask = cv2.inRange(lab, red_lower, red_upper)
    yellow_mask = cv2.inRange(img, lower_yellow, upper_yellow)
    
    #mask = cv2.medianBlur(yellow_mask, 5)

    # Combine the green and red masks
    #mask = cv2.bitwise_or(green_mask, red_mask)

    # Apply morphological operations to clean up the mask
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)

    # Apply Hough Transform to detect circles
    circles = cv2.HoughCircles(yellow_mask, cv2.HOUGH_GRADIENT, 1, 10,
                            param1=30, param2=15, minRadius=0, maxRadius=0)

    img_copy = img.copy()
    # Draw the circles
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            x, y, r = circle
            cv2.circle(img_copy, (x, y), r, (0, 0, 0), 2)
    
    # Display the result
    # cv2.imshow('Detected circles', img_copy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return circles
    
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
    
def rectangle_recognize(img):
    
    # Convert to RGB color space
    bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Define lower and upper bounds for yellow color in RGB color space
    lower_yellow = np.array([180, 180, 0])
    upper_yellow = np.array([255, 255, 100])

    # Threshold image to get yellow regions
    mask = cv2.inRange(bgr, lower_yellow, upper_yellow)

    # Find contours of yellow regions
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over contours to check if they are rectangles
    rectangles = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if aspect_ratio >= 0.9 and aspect_ratio <= 1.1:
                rectangles.append([cnt])
    
    
    img_copy = img.copy()  
    cv2.drawContours(img_copy, [cnt], 0, (0, 0, 0), 2)
    # cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 0), 2)
    # Show image with detected rectangles
    cv2.imshow('image', img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return rectangles

def video2matrix(img):
    #img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
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

    #根据tag的位置识别颜色
    #宽度
    



    # print(tag.corners)
    # cv2.imshow("apriltag_test",img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # sys.exit()
    return img


if __name__ == "__main__":
    # img = cv2.imread('./shape_recognize/test.png')
    # circle_recognize(img)
    # triangle_recognize(img)
    # rectangle_recognize(img)
    cap = cv2.VideoCapture('./video.mp4')
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter('video.mp4',fourcc,30,(960,720),True)

    # i = 0
    # while True:
    #     success, frame = cap.read()
    #     if success:
    #         i +=1
    #         print('i=',i)
    #         if(i>=100 and i<=300):
    #             out.write(frame)
    #     else:
    #         print('end')
    #         break

    while cap.isOpened():
        #get a frame
        ret, frame = cap.read()
        img = video2matrix(frame)

        cv2.imshow("capture", img)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    print('播放完毕')
    cap.release()
    cv2.destroyAllWindows()
    
