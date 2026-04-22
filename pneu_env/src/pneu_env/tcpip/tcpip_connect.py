import socket
import struct
import time
import json
import threading

stop = False
start_time = time.time()

def read_ctrl_file():
    while True:
        try:
            with open('ctrl.json', 'r') as f:
                ctrl_data = json.load(f)
            break
        except:
            continue
            # with open('ctrl_backup.json', 'r') as f:
            #     ctrl_data = json.load(f)
    
    return list(ctrl_data.values())

def write_obs_file(obs):
    obs_data = dict(
        time = obs[0],
        sen_pos = obs[1],
        sen_neg = obs[2],
        ref_pos = obs[3],
        ref_neg = obs[4],
        ctrl_pos = obs[5],
        ctrl_neg = obs[6]
    )
    with open('obs.json', 'w') as f:
        json.dump(obs_data, f)

def write_ctrl_file(obs):
    if len(obs) == 0:
        exit()
    obs_data = dict(
        time = obs[0],
        sen_pos = obs[1],
        sen_neg = obs[2],
        ref_pos = obs[3],
        ref_neg = obs[4],
        ctrl_pos = obs[5],
        ctrl_neg = obs[6]
    )
    with open('ctrl.json', 'w') as f:
        json.dump(obs_data, f)
        
def client_main():
    global stop
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('Wait for connect...')
    client.connect(('192.168.0.178', 8610))
    print('Connected!') 

    stop_flag = threading.Event()
    flag_time = time.time()
    try:
        while not stop_flag.is_set():
            ctrl_msg = read_ctrl_file()
            encoded_ctrl_msg = struct.pack('f'*7, *ctrl_msg)
            client.sendall(encoded_ctrl_msg)

            encoded_obs_msg = client.recv(7*4)
            if not encoded_obs_msg:
                break
            obs_msg = struct.unpack('f'*7, encoded_obs_msg)
            obs_msg = list(obs_msg)
            write_obs_file(obs_msg)
            
            if time.time() - flag_time > 1: 
                print(f'Received: {obs_msg} Time: {time.time() - start_time:.4f}')
                flag_time = time.time()
    except KeyboardInterrupt:
        print('Keyboad interrupt received. Stopping the loop')
        stop_flag.set()
    finally:
        client.close()
    
if __name__ == '__main__':
    client_main()
