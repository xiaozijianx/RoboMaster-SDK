import cv2
import numpy as np
import sys
import pupil_apriltags as apriltag     # for windows
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#from robomaster import robot
import json
import socket
import time
import threading

# IP_ADDRESS = '192.168.0.2'
# PORT0 = 12340
# PORT1 = 12341
# PORT2 = 12342
# FILENAME0 = './matrix0.json'
# FILENAME1 = './matrix1.json'
# FILENAME2 = './matrix2.json'

#计算颜色距离
def disfromcolor(BGR,BGR0):
    B = BGR[0] - BGR0[0]
    G = BGR[1] - BGR0[1]
    R = BGR[2] - BGR0[2]
    return B*B + G*G + R*R
    
        
#判断最近的颜色，返回0，0.5, 1
def nearcolor(BGR):
    Green_BGR = np.array([70,80,70])
    Red_BGR = np.array([45,80,190])
    Black_BGR = np.array([40,40,50])
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
        
#向json中写入文件
def writematrix2json(dir,matrix):
    matrix_list = matrix.tolist()
    with open(dir,'w') as f:
        json.dump(matrix_list,f)


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
    # print(type(tags))
    # print(tags)
    # print("%d apriltags have been detected."%len(tags))
    for tag in tags:
        cv2.circle(img, tuple(tag.corners[0].astype(int)), 4,(255,0,0), 2) # left-bottom
        cv2.circle(img, tuple(tag.corners[1].astype(int)), 4,(255,0,0), 2) # right-bottom
        cv2.circle(img, tuple(tag.corners[2].astype(int)), 4,(255,0,0), 2) # right-top
        cv2.circle(img, tuple(tag.corners[3].astype(int)), 4,(255,0,0), 2) # left-top
        #print(tag.corners[0,0])

    #sys.exit()
    for tag in tags:
        if tag.tag_id == tag_id:
            #根据tag的位置识别颜色
            #宽度
            xrange = tag.corners[2,0] - tag.corners[3,0]
            yrange = tag.corners[1,1] - tag.corners[2,1]
            xrange = xrange * 1.25
            yrange = yrange * 1.25
            #中心
            x0 = (tag.corners[2,0] + tag.corners[3,0])/2
            y0 = (tag.corners[1,1] + tag.corners[2,1])/2

            matrixc = np.zeros([observed_size,observed_size])
            for i in range(observed_size):
                for j in range(observed_size):
                    count1s = 0
                    count2s = 0
                    count3s = 0
                    #本格单独判断,上下左右采四个点
                    if i == observed_size//2 and j == observed_size//2:
                        x1 = x0
                        y1 = y0 - 0.7*yrange
                        x2 = x0 + 0.7*xrange
                        y2 = y0
                        x3 = x0
                        y3 = y0 + 0.7*yrange
                        x4 = x0 - 0.7*xrange
                        y4 = y0

                        #sample point
                        cv2.circle(img, (int(x1),int(y1)), 4,(40,232,225), 2) 
                        cv2.circle(img, (int(x2),int(y2)), 4,(40,232,225), 2) 
                        cv2.circle(img, (int(x3),int(y3)), 4,(40,232,225), 2) 
                        cv2.circle(img, (int(x4),int(y4)), 4,(40,232,225), 2) 

                        if x1 > img.shape[1]-1:
                            x1 = img.shape[1]-1
                        elif x1 < 0:
                            x1 = 0
                        
                        if y1 > img.shape[0]-1:
                            y1 = img.shape[0]-1
                        elif y1 < 0:
                            y1 = 0

                        if x2 > img.shape[1]-1:
                            x2 = img.shape[1]-1
                        elif x2 < 0:
                            x2 = 0
                        
                        if y2 > img.shape[0]-1:
                            y2 = img.shape[0]-1
                        elif y2 < 0:
                            y2 = 0
                        
                        if x3 > img.shape[1]-1:
                            x3 = img.shape[1]-1
                        elif x3 < 0:
                            x3 = 0
                        
                        if y3 > img.shape[0]-1:
                            y3 = img.shape[0]-1
                        elif y3 < 0:
                            y3 = 0
                        
                        if x4 > img.shape[1]-1:
                            x4 = img.shape[1]-1
                        elif x4 < 0:
                            x4 = 0
                        
                        if y4 > img.shape[0]-1:
                            y4 = img.shape[0]-1
                        elif y4 < 0:
                            y4 = 0

                        near = nearcolor(img[int(y1),int(x1)])
                        if near == 0.:
                            count1s = count1s + 1
                        elif near == 0.5:
                            count2s = count2s + 1
                        elif near == 1.0:
                            count3s = count3s + 1

                        near = nearcolor(img[int(y2),int(x2)])
                        if near == 0.:
                            count1s = count1s + 1
                        elif near == 0.5:
                            count2s = count2s + 1
                        elif near == 1.0:
                            count3s = count3s + 1

                        near = nearcolor(img[int(y3),int(x3)])
                        if near == 0.:
                            count1s = count1s + 1
                        elif near == 0.5:
                            count2s = count2s + 1
                        elif near == 1.0:
                            count3s = count3s + 1

                        near = nearcolor(img[int(y4),int(x4)])
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
                        x = x0 - 2 * xrange + 2 * j * xrange
                        y = y0 - 2 * yrange + 2 * i * yrange
                        x1 = x
                        y1 = y - 0.7*yrange
                        x2 = x + 0.7*xrange
                        y2 = y
                        x3 = x
                        y3 = y + 0.7*yrange
                        x4 = x - 0.7*xrange
                        y4 = y
                        #sample point
                        cv2.circle(img, (int(x1),int(y1)), 4,(40,232,225), 2)
                        cv2.circle(img, (int(x2),int(y2)), 4,(40,232,225), 2)
                        cv2.circle(img, (int(x3),int(y3)), 4,(40,232,225), 2)
                        cv2.circle(img, (int(x4),int(y4)), 4,(40,232,225), 2)

                        if x1 > img.shape[1]-1:
                            x1 = img.shape[1]-1
                        elif x1 < 0:
                            x1 = 0
                        
                        if y1 > img.shape[0]-1:
                            y1 = img.shape[0]-1
                        elif y1 < 0:
                            y1 = 0

                        if x2 > img.shape[1]-1:
                            x2 = img.shape[1]-1
                        elif x2 < 0:
                            x2 = 0
                        
                        if y2 > img.shape[0]-1:
                            y2 = img.shape[0]-1
                        elif y2 < 0:
                            y2 = 0
                        
                        if x3 > img.shape[1]-1:
                            x3 = img.shape[1]-1
                        elif x3 < 0:
                            x3 = 0
                        
                        if y3 > img.shape[0]-1:
                            y3 = img.shape[0]-1
                        elif y3 < 0:
                            y3 = 0
                        
                        if x4 > img.shape[1]-1:
                            x4 = img.shape[1]-1
                        elif x4 < 0:
                            x4 = 0
                        
                        if y4 > img.shape[0]-1:
                            y4 = img.shape[0]-1
                        elif y4 < 0:
                            y4 = 0

                        
                        near = nearcolor(img[int(y1),int(x1)])
                        if near == 0.:
                            count1s = count1s + 1
                        elif near == 0.5:
                            count2s = count2s + 1
                        elif near == 1.0:
                            count3s = count3s + 1

                        

                        near = nearcolor(img[int(y2),int(x2)])
                        if near == 0.:
                            count1s = count1s + 1
                        elif near == 0.5:
                            count2s = count2s + 1
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
                            count2s = count2s + 1
                        elif near == 1.0:
                            count3s = count3s + 1    

                        if count1s >= count2s:
                            if count1s >= count3s:
                                matrixc[i,j] = 0
                                text = str(int(matrixc[i,j]))
                                cv2.putText(img,text,(int(x),int(y)),cv2.FONT_HERSHEY_PLAIN,2.0,(255,255,255),1)
                            else:
                                matrixc[i,j] = 2
                                text = str(int(matrixc[i,j]))
                                cv2.putText(img,text,(int(x),int(y)),cv2.FONT_HERSHEY_PLAIN,2.0,(255,255,255),1)
                        else:
                            if count2s >= count3s:
                                matrixc[i,j] = 1
                                text = str(int(matrixc[i,j]))
                                cv2.putText(img,text,(int(x),int(y)),cv2.FONT_HERSHEY_PLAIN,2.0,(255,255,255),1)
                            else:
                                matrixc[i,j] = 2
                                text = str(int(matrixc[i,j]))
                                cv2.putText(img,text,(int(x),int(y)),cv2.FONT_HERSHEY_PLAIN,2.0,(255,255,255),1) 
            matrixname = "./matrix" + str(tag_id) + ".json"
            writematrix2json(matrixname, matrixc)
            # np.savetxt(matrixname, matrixc, fmt='%f')
    # print(tag.corners)
    # cv2.imshow("apriltag_test",img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # sys.exit()
    #sys.exit()
    return img

# #定义发送端线程函数
# def send_data(IP_ADDRESS, PORT, FILENAME):
#     while True:
#         #Read data from file
#         with open(FILENAME,'R') as f:
#             data = json.load(f)

#         #Serialize data to JSON and send over socket
#         serialized_data = json.dumps(data).encode()
#         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#             s.connect((IP_ADDRESS, PORT))
#             s.sendall(serialized_data)

#         #wait for some time before sending next update
#         time.sleep(1)


if __name__ == "__main__":
    # img = cv2.imread('./picture.png')
    # print(img.shape)
    # print(img[252,158])
    # print(img[278,68])
    # print(img[355,75])
 
    # #视频裁剪
    # cap = cv2.VideoCapture('./video_save.mp4')
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter('video.mp4',fourcc,30,(960,720),True)

    # i = 0
    # while True:
    #     success, frame = cap.read()
    #     if success:
    #         i +=1
    #         print('i=',i)
    #         if(i>=50 and i<=300):
    #             out.write(frame)
    #     else:
    #         print('end')
    #         break

    # sys.exit()

    #视频测试
    # #建立链接
    # send_thread0 = threading.Thread(target=send_data, args=(IP_ADDRESS, PORT0, FILENAME0,))
    # send_thread0.start()
    cap = cv2.VideoCapture('./video.mp4')

    while cap.isOpened():
        #get a frame
        ret, frame = cap.read()
        # #采集颜色并截图
        # print(frame[330,387])
        # print(frame[256,400])
        # print(frame[249,131])
        # print(frame[319,28])
        # cv2.imwrite('test.jpg',frame)
        # sys.exit()
        img1 = video2matrix(frame,0,3,3)
        # img2 = video2matrix(frame,1,3,3)
        # img3 = video2matrix(frame,2,3,3)
        cv2.imshow("capture", img1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print('播放完毕')
    cap.release()
    cv2.destroyAllWindows()
    # send_thread0.join()
    # #无人机飞行测试
    # tl_drone = robot.Drone()
    # tl_drone.initialize()

    # tl_camera = tl_drone.camera
    # # 显示302帧图传
    # tl_camera.start_video_stream(display=False)
    # tl_camera.set_fps("high")
    # tl_camera.set_resolution("high")
    # tl_camera.set_bitrate(6)

    # for i in range(0, 1000):
    #     img = tl_camera.read_cv2_image()
    #     # img_copy = triangle_recognize(img)
    #     #print(img.shape)
    #     img1 = video2matrix(img,0,3,3)

    #     # out.write(img)
    #     cv2.imshow("Drone", img1)
    #     cv2.waitKey(1)
    # cv2.destroyAllWindows()
    # tl_camera.stop_video_stream()

    # tl_drone.close()