import cv2
import numpy as np
import sys
import pupil_apriltags as apriltag     # for windows
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from robomaster import robot
import time
from client import writematrix2json

#计算颜色距离
def disfromcolor(BGR,BGR0):
    B = BGR[0] - BGR0[0]
    G = BGR[1] - BGR0[1]
    R = BGR[2] - BGR0[2]
    return B*B + G*G + R*R
    
        
#判断最近的颜色，返回0，0.5, 1
def nearcolor(BGR):
    Green_BGR = np.array([90,120,90])
    Red_BGR = np.array([40,70,190])
    Black_BGR = np.array([40,40,55])
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
        


def size_regular(img_shape,x,y):
    if x > img_shape[1]-1:
        x = img_shape[1]-1
    elif x < 0:
        x = 0

    if y > img_shape[0]-1:
        y = img_shape[0]-1
    elif y < 0:
        y = 0
    return x, y 

def counts_cal(count1s,count2s,count3s,x,y,img):
    near = nearcolor(img[int(y),int(x)])
    if near == 0.:
        count1s = count1s + 1
    elif near == 0.5:
        count2s = count2s + 1
    elif near == 1.0:
        count3s = count3s + 1

    return count1s,count2s,count3s

#判断颜色，适合有二维码的情况
def colorjudgementwithtag(img,x,y,xrange,yrange,coefficient,draw = False):
    x1 = x
    y1 = y - coefficient*yrange
    x2 = x + coefficient*xrange
    y2 = y
    x3 = x
    y3 = y + coefficient*yrange
    x4 = x - coefficient*xrange
    y4 = y
    
    x1,y1 = size_regular(img.shape,x1,y1)
    x2,y2 = size_regular(img.shape,x2,y2)
    x3,y3 = size_regular(img.shape,x3,y3)
    x4,y4 = size_regular(img.shape,x4,y4)

    count1s = 0
    count2s = 0
    count3s = 0

    count1s,count2s,count3s = counts_cal(count1s,count2s,count3s,x1,y1,img)
    count1s,count2s,count3s = counts_cal(count1s,count2s,count3s,x2,y2,img)
    count1s,count2s,count3s = counts_cal(count1s,count2s,count3s,x3,y3,img)
    count1s,count2s,count3s = counts_cal(count1s,count2s,count3s,x4,y4,img)

    if draw == True:
        #sample point
        cv2.circle(img, (int(x1),int(y1)), 4,(40,232,225), 2) 
        cv2.circle(img, (int(x2),int(y2)), 4,(40,232,225), 2) 
        cv2.circle(img, (int(x3),int(y3)), 4,(40,232,225), 2) 
        cv2.circle(img, (int(x4),int(y4)), 4,(40,232,225), 2) 
    
    if count1s >= count2s:
        if count1s >= count3s:
            return 0,img
        else:
            return 2,img
    else:
        if count2s >= count3s:
            return 1,img
        else:
            return 2,img

#判断颜色，适合无二维码的情况
def colorjudgementwithouttag(img,x,y,sample_size=3,draw = False):
    count1s = 0
    count2s = 0
    count3s = 0
    for m in range(sample_size):
        for n in range(sample_size):
            x_sample = x +8*(m - int((sample_size-1)//2))
            y_sample = y +8*(n - int((sample_size-1)//2))
            x_sample,y_sample = size_regular(img.shape,x_sample,y_sample)
            count1s,count2s,count3s = counts_cal(count1s,count2s,count3s,x_sample,y_sample,img)
 
    if draw == True:
        for m in range(sample_size):
            for n in range(sample_size):
                x_sample = x +10*(m - int((sample_size-1)//2))
                y_sample = y +10*(n - int((sample_size-1)//2))
                x_sample,y_sample = size_regular(img.shape,x_sample,y_sample)
                #sample point
                cv2.circle(img, (int(x_sample),int(y_sample)), 1,(40,232,225), 1) 

    if count1s >= count2s:
        if count1s >= count3s:
            return 0,img
        else:
            return 2,img
    else:
        if count2s >= count3s:
            return 1,img
        else:
            return 2,img

def videowithtag2matrix(img, tag_id, observed_size):
    matrixname = "./matrix.json"
    matrixc = np.zeros([observed_size,observed_size])
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
    if len(tags) >= 1:
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
                
                for i in range(observed_size):
                    for j in range(observed_size):
                        x = x0 - 2 * xrange + 2 * j * xrange
                        y = y0 - 2 * yrange + 2 * i * yrange
                        color,img = colorjudgementwithtag(img,x,y,xrange,yrange,0.7,True)
                        matrixc[i,j] = color
                        text = str(int(matrixc[i,j]))
                        cv2.putText(img,text,(int(x),int(y)),cv2.FONT_HERSHEY_PLAIN,2.0,(255,255,255),1)
                matrixname = "./matrix" + str(tag_id) + ".json"
    writematrix2json(matrixname, matrixc)
    return img, matrixname

def videowithouttag2matrix(img, uav_id,observed_size,  sample_size = 3, dis = 80):
    matrixname = "./matrix.json"
    matrixc = np.zeros([observed_size,observed_size])
    #无人机位置对应的中心 四点：右下 553 ,324右上554 ,151  左上 383 ，151
    #距离幕布 190cm， 投影宽度230cm 四点：右下 541 ,299右上 541 ,184  左上 430 ，184
    x0 = (430 + 541)/2
    y0 = (184 + 299)/2
    #一格的大小左下382,287 右下548， 289 右上551 117 左上381 115（x,y）
    xrange = 541 - 430
    yrange = 299 - 184
    # #投影画面的大小
    # x_size=300
    # y_size=300
    # #计算一格的大小
    # #利用距离tello的fov和距离计算，65.5 82.6上下对称
    # angle1_1 = (65.5/57.3)/2
    # angle1_2 = (65.5/57.3)/2
    # angle2_1 = (82.6/57.3)/2
    # angle2_2 = (82.6/57.3)/2
    # # print(dis * math.tan(angle2))
    # # sys.exit()
    # xrange = 2*img.shape[1]/(dis * math.tan(angle2_1)/x_size*10) #+ img.shape[1]/dis * math.tan(angle2_2)/x_size*10
    # yrange = 2*img.shape[0]/(dis * math.tan(angle1_1)/y_size*10) #+ img.shape[0]/dis * math.tan(angle1_2)/y_size*10

    for i in range(observed_size):
        for j in range(observed_size):
            x = x0 - xrange + j * xrange
            y = y0 - yrange + i * yrange
            color,img = colorjudgementwithouttag(img,x,y,sample_size,True)
            matrixc[i,j] = color
            text = str(int(matrixc[i,j]))
            cv2.putText(img,text,(int(x),int(y)),cv2.FONT_HERSHEY_PLAIN,2.0,(255,255,255),1)
    matrixname = "./matrix" + str(uav_id) + ".json"
    writematrix2json(matrixname, matrixc)
    return img, matrixname

if __name__ == "__main__":
    # img = cv2.imread('./picture1.png')
    # print(img.shape)
    # print(img[48,155])
    # print(img[106,136])
    # print(img[283,197])
    # print(img[352,205])
    # sys.exit()
 
    # #视频裁剪###########################################################
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

    #视频测试有tag#############################################
    # #建立链接
    # # Create threads for sending and receiving data
    # receive_thread0 = threading.Thread(target=receive_command, args=(filename0,PORT0,))
    # receive_thread1 = threading.Thread(target=receive_command, args=(filename1,PORT1,))
    # receive_thread2 = threading.Thread(target=receive_command, args=(filename2,PORT2,))

    # stop_sending = False
    # send_thread0 = threading.Thread(target=send_data, args=(IP_ADDRESS, PORT0, FILENAME0,))
    # send_thread0.start()
    # cap = cv2.VideoCapture('./video_save.mp4')

    # while cap.isOpened():
    #     #get a frame
    #     ret, frame = cap.read()
    #     # #采集颜色并截图
    #     # print(frame[330,387])
    #     # print(frame[256,400])
    #     # print(frame[249,131])
    #     # print(frame[319,28])
    #     # cv2.imwrite('test.jpg',frame)
    #     # sys.exit()
    #     if (frame is not None):
    #         img1,_ = videowithtag2matrix(frame,0,3)
    #         # img2 = videowithtag2matrix(frame,1,3)
    #         # img3 = videowithtag2matrix(frame,2,3)
    #         cv2.imshow("capture", img1)
    #     else:
    #         break
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # print('播放完毕')
    # cap.release()
    # cv2.destroyAllWindows()
    # send_thread0.join()


    #无人机飞行测试######################################################3
    tl_drone = robot.Drone()
    tl_drone.initialize()

    tl_camera = tl_drone.camera
    # 显示302帧图传
    tl_camera.start_video_stream(display=False)
    tl_camera.set_fps("high")
    tl_camera.set_resolution("high")
    tl_camera.set_bitrate(6)

    for i in range(0, 2000):
        img = tl_camera.read_cv2_image()
        # img_copy = triangle_recognize(img)
        #print(img.shape)
        img1,_ = videowithouttag2matrix(img,0,3,3)

        # out.write(img)
        cv2.imshow("Drone", img1)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    tl_camera.stop_video_stream()

    tl_drone.close()


    # #视频测试无tag,也可以截图############################################3
    # # #建立链接
    # # send_thread0 = threading.Thread(target=send_data, args=(IP_ADDRESS, PORT0, FILENAME0,))
    # # send_thread0.start()
    # cap = cv2.VideoCapture('./video_save.mp4')
    # i = 0
    # while cap.isOpened():
    #     #get a frame
    #     ret, frame = cap.read()
    #     # #采集颜色并截图
    #     # print(frame[330,387])
    #     # print(frame[256,400])
    #     # print(frame[249,131])
    #     # print(frame[319,28])
    #     # cv2.imwrite('test.jpg',frame)
    #     # sys.exit()
    #     # if ret:
    #     #     i +=1
    #     #     print('i=',i)
    #     #     if i == 200:
    #     #         comb_file = './test/picture1.png'
    #     #         cv2.imencode('.png', frame)[1].tofile(comb_file)
    #     #         sys.exit()
    #     if (frame is not None):   
    #         img1,_ = videowithouttag2matrix(frame,0,3,3)
    #         # img2 = videowithtag2matrix(frame,1,3)
    #         # img3 = videowithtag2matrix(frame,2,3)
    #         cv2.imshow("capture", img1)
    #     else:
    #         break
    #     if cv2.waitKey(30) & 0xFF == ord('q'):
    #         break
    # print('播放完毕')
    # cap.release()
    # cv2.destroyAllWindows()
    # # send_thread0.join()