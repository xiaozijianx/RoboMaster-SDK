import robomaster
from robomaster import robot
import keyboard  #监听键盘

def forward(tl,dis):
    print("forward")
    tl.forward(distance=dis).wait_for_completed()

def backward(tl,dis):
    print("backward")
    tl.backward(distance=dis).wait_for_completed()

def up(tl,dis):
    print("up")
    tl.up(distance=dis).wait_for_completed()

def down(tl,dis):
    print("down")
    tl.down(distance=dis).wait_for_completed()

def left(tl,dis):
    print("left")
    tl.left(distance=dis).wait_for_completed()

def right(tl,dis):
    print("right")
    tl.right(distance=dis).wait_for_completed()

def take_off(tl):
    print("take_off")
    tl.takeoff().wait_for_completed()

def land(tl):
    print("land")
    tl.land().wait_for_completed()

if __name__ == '__main__':
    # 如果本地IP 自动获取不正确，手动指定本地IP地址
    #robomaster.config.LOCAL_IP_STR = "192.168.10.2"
    tl_drone = robot.Drone()
    # 初始化
    tl_drone.initialize()
    print("初始化完毕")

    tl_flight = tl_drone.flight

    keyboard.add_hotkey('up', up,  args=(tl_flight,20,))
    keyboard.add_hotkey('down', down,  args=(tl_flight,20,))
    keyboard.add_hotkey('left', left,  args=(tl_flight,20,))
    keyboard.add_hotkey('right', right,  args=(tl_flight,20,))

    keyboard.add_hotkey('e', take_off,  args=(tl_flight,))
    keyboard.add_hotkey('q', land,  args=(tl_flight,))

    #前后
    keyboard.add_hotkey('w', forward,  args=(tl_flight,20,))
    keyboard.add_hotkey('s', backward,  args=(tl_flight,20,))
    
    
    #按ctrl+alt输出b
    keyboard.wait('space')
    #wait里也可以设置按键，说明当按到该键时结束

    tl_drone.close()