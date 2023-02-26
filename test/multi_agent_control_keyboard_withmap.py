from multi_robomaster import multi_robot
import keyboard  #监听键盘

def rc_forward(tl,b):
    print("rc_forward")
    tl.rc(a=0, b=b, c=0, d=0).wait_for_completed()

def rc_left(tl,a):
    print("rc_forward")
    tl.rc(a=a, b=0, c=0, d=0).wait_for_completed()

def forward(tl,dis):
    print("forward")
    tl.forward(distance=dis[0]).wait_for_completed()

def backward(tl,dis):
    print("backward")
    tl.backward(distance=dis[0]).wait_for_completed()

def up(tl,dis):
    print("up")
    tl.up(distance=dis[0]).wait_for_completed()

def down(tl,dis):
    print("down")
    tl.down(distance=dis[0]).wait_for_completed()

def left(tl,dis):
    print("left")
    tl.left(distance=dis[0]).wait_for_completed()

def right(tl,dis):
    print("right")
    tl.right(distance=dis[0]).wait_for_completed()

def take_off(tl,dis):
    print("take_off")
    tl.takeoff().wait_for_completed()

def land(tl,dis):
    print("land")
    tl.land().wait_for_completed()

def go(tl,dis):
    print("go")
    tl.land().wait_for_completed()

def group_forward(multi_drone,tello_group,dis):
    multi_drone.run([tello_group, forward, [20]])
    #tello_group.forward(distance=dis).wait_for_completed()

def group_backward(multi_drone,tello_group,dis):
    multi_drone.run([tello_group, backward, [20]])
    #tello_group.backward(distance=dis).wait_for_completed()

def group_up(multi_drone,tello_group,dis):
    multi_drone.run({[tello_group, up, [20]],[tello_group, up, [20]]})
    #tello_group.up(distance=dis).wait_for_completed()

def group_down(multi_drone,tello_group,dis):
    multi_drone.run([tello_group, down, [20]])
    #tello_group.down(distance=dis).wait_for_completed()


def group_left(multi_drone,tello_group,dis):
    multi_drone.run([tello_group, left, [dis]])
    #tello_group.left(distance=dis).wait_for_completed()


def group_right(multi_drone,tello_group,dis):
    multi_drone.run([tello_group, right, [dis]])
    #tello_group.right(distance=dis).wait_for_completed()

def group_take_off(multi_drone,tello_group):
    multi_drone.run([tello_group, take_off, [20]])
    #tello_group.takeoff().wait_for_completed()


def group_land(multi_drone,tello_group):
    multi_drone.run([tello_group, land, [20]])
    #tello_group.land().wait_for_completed()

def group_go(multi_drone,tello_group,dis):
    multi_drone.run([tello_group, go, [dis]])
    #tello_group.land().wait_for_completed()

if __name__ == '__main__':
    #Initializa
    multi_drone = multi_robot.MultiDrone()
    multi_drone.initialize(robot_num=2)
    print("初始化完毕")
    #scan ip and sn
    # drone_ip_list = multi_drone._get_sn(timeout=10)
    # for sn in drone_ip_list:
    #     print("scan result: sn:{0}, ip:{1}".format(sn, drone_ip_list[sn]))

    # # scan result: sn:0TQZK1FCNT1NFT, ip:('192.168.1.104', 8889)
    # sn_list = list(drone_ip_list.keys())

    # #可以通过手动输入来准确控制想要操控的无人机
    # robot_sn_list = [sn_list[0],sn_list[1]]
    robot_sn_list = ["0TQZK1FCNT1NAT", "0TQZK1HCNT1PLV"]
    #sn_list = []
    #battery_list = []
    drone_num = 2

    multi_drone.number_id_by_sn([0, robot_sn_list[0]],[1, robot_sn_list[1]])
    tello_group1 = multi_drone.build_group([0,1])
    tello_group1.mission_pad_on()
   
    #群组单机控制
    #通过sn码识别飞机
    drone_obj = tello_group1.get_robot(0)
    tl_flight = drone_obj.flight

    #群组集体控制
    keyboard.add_hotkey('up', group_go,  args=(multi_drone,tello_group1,50,))
    keyboard.add_hotkey('down', group_down,  args=(multi_drone,tello_group1,20,))
    keyboard.add_hotkey('left', group_left,  args=(multi_drone,tello_group1,20,))
    keyboard.add_hotkey('right', group_right,  args=(multi_drone,tello_group1,20,))

    keyboard.add_hotkey('e', group_take_off,  args=(multi_drone,tello_group1,))
    keyboard.add_hotkey('q', group_land,  args=(multi_drone,tello_group1,))

    #前后
    keyboard.add_hotkey('w', group_forward,  args=(multi_drone,tello_group1,20,))
    keyboard.add_hotkey('s', group_backward,  args=(multi_drone,tello_group1,20,))
    #multi_drone.run([tello_group1, basic_task])

    keyboard.wait('space')
    tello_group1.mission_pad_off()
    multi_drone.close()