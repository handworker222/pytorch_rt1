import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from PIL import Image
import requests
import socket
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import json
from collections import deque

import torch
from maruya24_rt1.transformer_network import TransformerNetwork
from maruya24_rt1.transformer_network_test_set_up import state_space_list
from maruya24_rt1.tokenizers.utils import batched_space_sampler
from maruya24_rt1.tokenizers.utils import np_to_tensor

import util.misc as utils

import tensorflow_hub as hub
import matplotlib.image

class RT1Server:
    def __init__(self):
        
        with open("train_config.json", "r") as f:
            self.args = json.load(f)
        
        self.resume_from_checkpoint = "/workspace/rt1_torch/logs/1711272087/49-p24-checkpoint.pth"

        checkpoint = torch.load(self.resume_from_checkpoint, map_location="cpu")
        self._action_space = checkpoint["action_space"]
        network_configs = self.args["network_configs"]
        network_configs["time_sequence_length"] = self.args["time_sequence_length"]
        network_configs["num_encoders"] = len(self.args["cam_view"])
        network_configs["using_proprioception"] = self.args["using_proprioception"]
        network_configs["token_embedding_size"] = network_configs[
            "token_embedding_size_per_image"
        ] * len(self.args["cam_view"])
        del network_configs["token_embedding_size_per_image"]
        network_configs["input_tensor_space"] = state_space_list()[0]
        network_configs["output_tensor_space"] = self._action_space
        self.network = TransformerNetwork(**network_configs)
        try:
            local_rank = os.environ["LOCAL_RANK"]
            torch.cuda.set_device(int(local_rank))
        except:
            pass
        self.device = torch.device("cuda:0")
        self.network.to(self.device)
        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.network.eval()
        
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

        # vb.0.0 front + wrist view with math delta, multi obj+single obj mixed scene, octo base fine-tune
        # self.model = OctoModel.load_pretrained(checkpoints["vb.0.0_ma"][0],step=50000)
        
        
        self.task_text = ""
        self.his_img = deque(maxlen=1)
        self.his_img_wrist = deque(maxlen=1)

        # self.octo_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.octo_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # self.octo_server.bind(("0.0.0.0", 9000))
        # self.octo_server.listen(1)

        self.hello = "hello"
        self.ack = "ready"
        self.ack_end = "end"

    def server_close(self):
        self.rt1_server.close()

    def get_msg_and_send_action(self,sock):
        
        rec_hello = sock.recv(128)
        if len(rec_hello) == 0:
            print("recv data is none")
        else:
            if rec_hello.decode() == self.hello:
                print("hello recved!,wait for msg lens")
                sock.sendall(self.ack.encode())
                recv_len_msg = sock.recv(1024)
                recv_len_msg = json.loads(recv_len_msg.decode())

                if "len" in recv_len_msg:
                    print("get msg lens, ready for rec full msg")
                    r_len = recv_len_msg["len"]
                    sock.sendall(str(r_len).encode())

                    r_msg = ""
                    get = 0
                    while get < r_len:
                        buf_r = sock.recv(1024)
                        r_msg += buf_r.decode()
                        get +=len(buf_r)
                    
                    print("full msg received, prepare to predict actions")

                    full_msg = json.loads(r_msg)
                    sock.sendall(self.ack_end.encode())

                    action = self.rt1_inference(full_msg,has_wrist=False)
                    # print("action:",action)

                    action_msg = json.dumps({"action":action.tolist()})
                    len_action_msg = json.dumps({"len":len(action_msg)})

                    print("get predicted actions, ask for sending action msg")
                    sock.sendall(len_action_msg.encode())
                    action_len_back = sock.recv(1024)
                    if int(action_len_back.decode()) == len(action_msg):
                        print("client is ready,sending action msg")
                        sock.sendall(action_msg.encode())
                        action_send_back = sock.recv(1024)
                        if action_send_back.decode() == self.ack_end:
                            print("finish predicting action")
                            
                            return True
                        else:
                            print("recv action msg ERROR!")
                            return False
                    else:
                        print("recv action msg lens ERROR!")
                        return False

                else:
                    print("get msg lens error!")
                    return False
        

    def rt1_inference(self,msg,has_wrist):
        task_text = msg["task"]
        img_primary = np.array(msg["img_primary"],dtype=np.float32)
        
        img_primary /= 255.0
        matplotlib.image.imsave('/workspace/rt1_torch/figure1.png', img_primary)
        # img_wrist = np.array(msg["img_wrist"],dtype=np.uint8)
        cam_views = ['front', 'wrist'] if has_wrist else ['front']
        
        network_state = batched_space_sampler(self.network._state_space, batch_size=1)
        network_state = np_to_tensor(network_state)  # Convert np.ndarray to tensor
        for k, v in network_state.items():
            network_state[k] = torch.zeros_like(v).to(self.device)
        
        output_actions = []
        print("task text:",self.task_text)
        if task_text == self.task_text:
            print(111)
            if len(self.his_img)>0:
                self.imgs.pop(0)
                self.imgs.append(torch.from_numpy(img_primary).permute(2, 0, 1).unsqueeze(0))
                input_primary_imgs = torch.stack(self.imgs, dim=1).to(self.device)
                self.langs.pop(0)
                self.langs.append(self.langs_embed)
                input_language_embedding = torch.stack(self.langs, dim=1).to(self.device)
            else:
                self.imgs = [torch.zeros(1, 3, 300 * len(cam_views), 300)] * self.network._time_sequence_length
                self.imgs.pop(0)
                self.imgs.append(torch.from_numpy(img_primary).permute(2, 0, 1).unsqueeze(0))
                input_primary_imgs = torch.stack(self.imgs, dim=1).to(self.device)
                self.langs = [torch.zeros(1, self.network._language_embedding_size)] * self.network._time_sequence_length
                self.langs.pop(0)
                self.langs.append(self.langs_embed)
                input_language_embedding = torch.stack(self.langs, dim=1).to(self.device)
                
            if has_wrist:
                raise Exception("There should not be a wrist camera.")
                # observation = {
                #     'image_primary': input_primary_imgs,
                #     "image_wrist":input_wrist_imgs,
                #     'pad_mask': pad_mask}
            else:
                observation = {
                    'image': input_primary_imgs,
                    # "image_wrist":input_wrist_imgs,
                    'natural_language_embedding': input_language_embedding,}

            with torch.no_grad():
                # actions, network_state = self.network(observation, network_state)
                for i_ts in range(self.network._time_sequence_length):
                    ob = utils.retrieve_single_timestep(observation, i_ts)
                    output_action, network_state = self.network(ob, network_state)
                    output_actions.append(output_action)

                # Retrieve the final inferred action
                actions = output_actions[-1]
            
            actions = utils.dict_to_device(actions, torch.device('cpu'))
            
            print("action:",actions)

            self.his_img.append(img_primary)
            # self.his_img_wrist.append(img_wrist)
            actions = np.concatenate(
                (actions["world_vector"].flatten().numpy(),
                actions["rotation_delta"].flatten().numpy(),
                actions["gripper_closedness_action"].flatten().numpy(),
                actions["terminate_episode"].flatten().numpy()), axis=None
            )
            return actions


        else:
            print(222)
            self.task_text = task_text
            self.his_img = deque(maxlen=1)
            # self.his_img_wrist = deque(maxlen=1)

            self.imgs = [torch.zeros(1, 3, 300 * len(cam_views), 300)] * self.network._time_sequence_length
            self.imgs.pop(0)
            self.imgs.append(torch.from_numpy(img_primary).permute(2, 0, 1).unsqueeze(0))
            input_primary_imgs = torch.stack(self.imgs, dim=1).to(self.device)

            task = self.task_text.replace('_', '')
            self.langs_embed = torch.tensor(self._embed(tf.constant([task])).numpy())
            self.langs = [torch.zeros(1, self.network._language_embedding_size)] * self.network._time_sequence_length
            self.langs.pop(0)
            self.langs.append(self.langs_embed)
            input_language_embedding = torch.stack(self.langs, dim=1).to(self.device)

            if has_wrist:
                raise Exception("There should not be a wrist camera.")
                # observation = {
                #     'image_primary': input_primary_imgs,
                #     "image_wrist":input_wrist_imgs,
                #     'pad_mask': pad_mask}
            else:
                observation = {
                    'image': input_primary_imgs,
                    # "image_wrist":input_wrist_imgs,
                    'natural_language_embedding': input_language_embedding,}

            with torch.no_grad():
                # actions, network_state = self.network(observation, network_state)
                for i_ts in range(self.network._time_sequence_length):
                    ob = utils.retrieve_single_timestep(observation, i_ts)
                    output_action, network_state = self.network(ob, network_state)
                    output_actions.append(output_action)

                # Retrieve the final inferred action
                actions = output_actions[-1]
            
            actions = utils.dict_to_device(actions, torch.device('cpu'))
            
            print("action:",actions)

            self.his_img.append(img_primary)
            # self.his_img_wrist.append(img_wrist)
            actions = np.concatenate(
                (actions["world_vector"].flatten().numpy(),
                actions["rotation_delta"].flatten().numpy(),
                actions["gripper_closedness_action"].flatten().numpy(),
                actions["terminate_episode"].flatten().numpy()), axis=None
            )
            return actions





if __name__ == "__main__":
    os = RT1Server()
    rt1_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    rt1_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    rt1_server.bind(("0.0.0.0", 9001))
    rt1_server.listen(2)

    # os = OctoServer(octo_server)

    while True:
        print("waiting for connection")
        sock,addr = rt1_server.accept()
        sock.settimeout(300.0)
        while True:
            print("client connected, start model inference:")
            # is_ok = os.get_msg_and_send_action(sock)
            try:
                is_ok = os.get_msg_and_send_action(sock)
                if is_ok:
                    print("finish one step!")
                    
                else:
                    print("server error!")
                    break
            except:
                print("time out")
                break

   
    # while True:
    #     is_ok = os.get_msg_and_send_action()
    #     if is_ok:
    #         print("finish one step!")
    #         break
    #     else:
    #         print("server error!")
    #         break
