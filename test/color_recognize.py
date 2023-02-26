import cv2
import numpy as np
import sys
import pupil_apriltags as apriltag     # for windows
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#计算颜色距离
def disfromcolor(BGR,BGR0):
    B = BGR[0] - BGR0[0]
    G = BGR[1] - BGR0[1]
    R = BGR[2] - BGR0[2]
    return B*B + G*G + R*R
    
        
#判断最近的颜色，返回0，0.5, 1
def nearcolor(BGR):
    Green_BGR = np.array([80,110,25])
    Red_BGR = np.array([25,85,230])
    Black_BGR = np.array([20,20,20])
    dis1 = disfromcolor(BGR, Green_BGR)
    dis2 = disfromcolor(BGR, Red_BGR)
    dis3 = disfromcolor(BGR, Black_BGR)
    if dis1 <= dis2:
        if dis1 <= dis3:
            return 0.
        else:
            return 1.
    else:
        if dis2 <= dis3:
            return 0.5
        else:
            return 1.

def video2matrix(img, tag_id, observed_size, a):
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
        #print(tag.corners[0,0])

    #sys.exit()
    for tag in tags:
        if tag.tag_id == tag_id:
            #根据tag的位置识别颜色
            #宽度
            xrange = tag.corners[1,0] - tag.corners[0,0]
            yrange = tag.corners[2,1] - tag.corners[1,1]
            #中心
            x0 = (tag.corners[1,0] + tag.corners[0,0])/2
            y0 = (tag.corners[2,1] + tag.corners[1,1])/2

            matrixc = np.zeros([observed_size,observed_size])
            for i in range(observed_size):
                for j in range(observed_size):
                    count1s = 0
                    count2s = 0
                    count3s = 0
                    #本格单独判断,上下左右采四个点
                    if i == observed_size//2 and j == observed_size//2:
                        x1 = x0
                        y1 = y0 - yrange
                        x2 = x0 + xrange
                        y2 = y0
                        x3 = x0
                        y3 = y0 + yrange
                        x4 = x0 - xrange
                        y4 = y0

                        near = nearcolor(img[int(y1),int(x1)])
                        if near == 0.:
                            count1s = count1s + 1
                        elif near == 0.5:
                            count2s + count2s + 1
                        elif near == 1.0:
                            count3s = count3s + 1

                        near = nearcolor(img[int(y2),int(x2)])
                        if near == 0.:
                            count1s = count1s + 1
                        elif near == 0.5:
                            count2s + count2s + 1
                        elif near == 1.0:
                            count3s = count3s + 1

                        near = nearcolor(img[int(y3),int(x3)])
                        if near == 0.:
                            count1s = count1s + 1
                        elif near == 0.5:
                            count2s + count2s + 1
                        elif near == 1.0:
                            count3s = count3s + 1

                        near = nearcolor(img[int(y4),int(x4)])
                        if near == 0.:
                            count1s = count1s + 1
                        elif near == 0.5:
                            count2s + count2s + 1
                        elif near == 1.0:
                            count3s = count3s + 1    
                        
                        if count1s >= count2s:
                            if count1s >= count3s:
                                matrixc[i,j] = 0
                                text = str(int(matrixc[i,j]))
                                cv2.putText(img,text,(int(x0),int(y0)),cv2.FONT_HERSHEY_PLAIN,2.0,(0,0,255),1)
                            else:
                                matrixc[i,j] = 2
                                text = str(int(matrixc[i,j]))
                                cv2.putText(img,text,(int(x0),int(y0)),cv2.FONT_HERSHEY_PLAIN,2.0,(0,0,255),1)
                        else:
                            if count2s >= count3s:
                                matrixc[i,j] = 1
                                text = str(int(matrixc[i,j]))
                                cv2.putText(img,text,(int(x0),int(y0)),cv2.FONT_HERSHEY_PLAIN,2.0,(0,0,255),1)
                            else:
                                matrixc[i,j] = 2
                                text = str(int(matrixc[i,j]))
                                cv2.putText(img,text,(int(x0),int(y0)),cv2.FONT_HERSHEY_PLAIN,2.0,(0,0,255),1) 
                    else:#然后判断四周
                        x = x0 - xrange + i * xrange
                        y = y0 - yrange + j * yrange
                        for m in range(a):
                            for n in range(a):
                                x_sample = x+m-int((a-1)/2)
                                y_sample = y+n-int((a-1)/2)
                                if x_sample > 959:
                                    x_sample = 959
                                elif x_sample < 0:
                                    x_sample = 0
                                
                                if y_sample > 719:
                                    y_sample = 719
                                elif y_sample < 0:
                                    y_sample = 0
                                near = nearcolor(img[int(y_sample),int(x_sample)])
                                if near == 0.:
                                    count1s = count1s + 1
                                elif near == 0.5:
                                    count2s = count2s + 1
                                elif near == 1.0:
                                    count3s = count3s + 1
                                
                                if count1s >= count2s:
                                    if count1s >= count3s:
                                        matrixc[i,j] = 0
                                        text = str(int(matrixc[i,j]))
                                        cv2.putText(img,text,(int(x),int(y)),cv2.FONT_HERSHEY_PLAIN,2.0,(0,0,255),1)
                                    else:
                                        matrixc[i,j] = 2
                                        text = str(int(matrixc[i,j]))
                                        cv2.putText(img,text,(int(x),int(y)),cv2.FONT_HERSHEY_PLAIN,2.0,(0,0,255),1)
                                else:
                                    if count2s >= count3s:
                                        matrixc[i,j] = 1
                                        text = str(int(matrixc[i,j]))
                                        cv2.putText(img,text,(int(x),int(y)),cv2.FONT_HERSHEY_PLAIN,2.0,(0,0,255),1)
                                    else:
                                        matrixc[i,j] = 2
                                        text = str(int(matrixc[i,j]))
                                        cv2.putText(img,text,(int(x),int(y)),cv2.FONT_HERSHEY_PLAIN,2.0,(0,0,255),1) 
            matrixname = "./matrix" + str(tag_id) + ".txt"

            np.savetxt(matrixname, matrixc, fmt='%f')
    # print(tag.corners)
    # cv2.imshow("apriltag_test",img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # sys.exit()
    #sys.exit()
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
        img1 = video2matrix(frame,0,3,3)
        # img2 = video2matrix(frame,0,3,3)
        # img3 = video2matrix(frame,0,3,3)

        cv2.imshow("capture", img1)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    print('播放完毕')
    cap.release()
    cv2.destroyAllWindows()
    
