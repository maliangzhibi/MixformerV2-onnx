import socket
import json

if __name__=="__main__":
    id = 0
    while True:
        # 创建UDP套接字
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # 目标服务器的IP和端口
        server_address = ('localhost', 12345)

        # 要发送的JSON数据
        data = {
            'name': 'Alice',
            'age': id
        }
        id += 1
        message = json.dumps(data).encode('utf-8')
        
        # 发送数据
        sock.sendto(message, server_address)
        sock.close()