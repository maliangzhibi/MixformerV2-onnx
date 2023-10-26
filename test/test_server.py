import socket
import json

# 创建UDP套接字
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = ('localhost', 12345)
sock.bind(server_address)

while True:
    # 等待数据
    data, address = sock.recvfrom(4096)
    
    # 解析JSON数据
    received_data = json.loads(data.decode('utf-8'))
    print("Received data:", received_data)

    # 如果你不想无限期地等待，可以加入退出条件
    if received_data.get("exit"):
        break

sock.close()