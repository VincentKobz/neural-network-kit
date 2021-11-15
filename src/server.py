import socket
import sys
import threading

def main(port):
    print("Server starting ...")
    start_server(port)

def communicate(client_socket):
    client_socket.send(str.encode("Connection established"))
    request = client_socket.recv(2048)
    if not request:
        client_socket.close()
    print("Client disconnected ...")

def start_server(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sck:
        sck.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sck.bind((socket.gethostname(), int(port)))
        sck.listen(5)
        print("Server running  ...")

        while True:
            (client_socket, address) = sck.accept()
            print("Client connected ...")
            new_thread = threading.Thread(target=communicate, args=client_socket)
            new_thread.daemon
            new_thread.start

if __name__ == "__main__":
    main(sys.argv[1])
