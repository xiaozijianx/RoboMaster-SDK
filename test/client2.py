import socket
import sys
import time
import threading
import json

IP_ADDRESS = '127.0.0.1'
PORT1 = 12340
PORT2 = 12341
PORT3 = 12342
FILENAME1 = './matrix1.json'
FILENAME2 = './matrix2.json'
FILENAME3 = './matrix3.json'

stop_receiving = False
stop_sending = False
COMMAND = [0,0,0,1,1]
client_socket = ''

mutex = threading.Lock()

#向json中写入文件
def writematrix2json(dir,matrix):
    mutex.acquire()
    matrix_list = matrix.tolist()
    with open(dir,'w') as f:
        json.dump(matrix_list,f)
    mutex.release()

#定义发送端线程函数
def send_data():
    global stop_sending
    global client_socket
    while not stop_sending:
        #Read data from file
        mutex.acquire()
        with open(FILENAME2,'r') as f:
            data = json.load(f)
            #Serialize data to JSON and send over socket
            serialized_data = json.dumps(data).encode()
            print(serialized_data)
            client_socket.sendall(serialized_data)
        mutex.release()
        #wait for some time before sending next update
        time.sleep(1)

#定义指令接收端函数
#接收文件的进程
def receive_command():
    global COMMAND
    global stop_receiving
    global client_socket
    while not stop_receiving:
        while True:
            received_data = client_socket.recv(1024)
            if not received_data:
                break
            #print('Received data:', received_data.decode())
            print(f"從伺服器收到訊息：{received_data.decode()}")
            if received_data.decode() == 'STOP':
                stop_threading()
                time.sleep(0.1)
                client_socket.send('STOPRECEIVED'.encode())
                client_socket.close()
                break
            command = received_data.decode().strip('[]').split(',')
            for index,i in enumerate(command):
                COMMAND[index] = float(i)

#停止线程
def stop_threading():
    global stop_sending
    global stop_receiving
    stop_sending = True
    stop_receiving = True

#返回收到的指令
def return_command():
    global COMMAND
    command1 = [0,0,0,0]
    command2 = 0
    command1 = COMMAND[0:3]
    command2 = COMMAND[4]
    return command1,command2

def client_establish(IP,PORT):
    global client_socket
    # 建立 Socket 物件
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 連線到伺服器
    client_socket.connect((IP, PORT))

if __name__ == "__main__":
    client_establish(IP_ADDRESS,PORT2)
    send_thread = threading.Thread(target=send_data)
    receive_thread = threading.Thread(target=receive_command)
    send_thread.start()
    receive_thread.start()
    # time.sleep(100)
    # stop_threading()
    # send_thread.join()
    # receive_thread.join()