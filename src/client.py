import socket
import sys

def client(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sck:
        sck.connect((socket.gethostname(), int(port)))
        print("Connected ...")
        while True:
            response = sck.recv(2048)
            if not response:
                break;
            print(response)
        sck.close()
        print("Disconnected ...")


def main(port):
    client(port)

if __name__ == '__main__':
    main(sys.argv[1])
