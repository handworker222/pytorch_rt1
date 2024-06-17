import os
import json
import glob
from PIL import Image
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import tensorflow_datasets as tfds

from tqdm import tqdm as tqdm

# This is the very first version for the tfrecord dataset, which is loading all once (consuming lots of memories).

def build_dataset(
    data_path,
    time_sequence_length=6,
    predicting_next_ts=True,
    num_train_episode=200,
    num_val_episode=100,
    cam_view=["front"],
    language_embedding_size=512,
    piece_num=0,
    es_streams=None,
):

    cam_dict = {
        "front": 0,
        "fronttop": 1,
        "root": 2,
        "root_shoulder": 3,
        "side": 4,
        "topdown": 5,
        "wrist": 6,
    }
    
    if piece_num == 0:
        b_streams = tfds.builder_from_directory(data_path)
        ds_streams = b_streams.as_dataset(split="train")
        es_streams = iter(ds_streams)
    episodes = []
    episode_length = []
    for _ in range(100*piece_num, min(100*(piece_num+1), 2492)): # len(ds_streams)
        es_stream = next(es_streams)
        t_streams = [step for step in es_stream["steps"]]
        episodes.append(t_streams)
        episode_length.append(len(t_streams))
    
    perm_indice = torch.randperm(len(episodes)).tolist()
    dirs_lengths = dict(
        episodes=np.array(episodes)[perm_indice],
        episode_length=np.array(episode_length)[perm_indice],
    )
    train_episodes = dirs_lengths["episodes"][:num_train_episode]
    train_episode_length = dirs_lengths["episode_length"][:num_train_episode]
    val_episodes = dirs_lengths["episodes"][
        num_train_episode : num_train_episode + num_val_episode
    ]
    val_episode_length = dirs_lengths["episode_length"][
        num_train_episode : num_train_episode + num_val_episode
    ]

    train_dataset = IODataset(
        episodes=train_episodes,
        episode_length=train_episode_length,
        time_sequence_length=time_sequence_length,
        predicting_next_ts=predicting_next_ts,
        cam_view=cam_view,
        cam_dict=cam_dict,
        language_embedding_size=language_embedding_size,
    )
    val_dataset = IODataset(
        episodes=val_episodes,
        episode_length=val_episode_length,
        time_sequence_length=time_sequence_length,
        predicting_next_ts=predicting_next_ts,
        cam_view=cam_view,
        cam_dict=cam_dict,
        language_embedding_size=language_embedding_size,
    )
    return train_dataset, val_dataset, es_streams


class IODataset(Dataset):
    def __init__(
        self,
        episodes,
        episode_length,
        time_sequence_length=6,
        predicting_next_ts=True,
        cam_view=["front"],
        cam_dict=None,
        robot_dof=9,
        language_embedding_size=512,
    ):
        self._cam_view = cam_view
        self._cam_dict = cam_dict
        self.predicting_next_ts = predicting_next_ts
        self._time_sequence_length = time_sequence_length
        self._episode_length = episode_length
        self.querys = self.generate_history_steps(episode_length)
        self._episodes = episodes
        self.values, self.num_zero_history_list = self.organize_file_names()
        self._robot_dof = robot_dof
        self._language_embedding_size = language_embedding_size
    
    def generate_history_steps(self, episode_length):
        """
        This function generates the step for current frame and history frames
        Parameters:
        - episode_length(list of int): number of episode lengths for each episode
        Returns:
        - keys(list of tensors): history steps for each data
        """
        querys = []
        for el in episode_length:
            q = torch.cat(
                (
                    [
                        torch.arange(el)[:, None] - i
                        for i in range(self._time_sequence_length)
                    ]
                ),
                dim=1,
            )
            q[q < 0] = -1
            querys.append(q.flip(1))
        return querys

    def organize_file_names(self):
        """
        This function generates the infor for each data, including how many zeros were padded
        data's episode directory, image filenames, and all the other parameters for data
        Parameters:
        -
        Returns:
        - values(list): each value including
            - num_zero_history: when we read at initial frames of a episode, it doesn't have history,
                then we need to pad zeros to make sure these aligns to data with history frames.
                this number specified how many frames of zero is padded
            - episode_dir: the episode directory where this data is stored
            - img_fns = img_fns: the images this data should read]
            - query_index = index of this data in this episode
            - episode_length = total length of this episode
        """
        values = []
        num_zero_history_list = []
        # for i, (query, key_img, ed) in enumerate(
        #     zip(self.querys, self.keys_image, self._episodes)
        # ):
        for i, query in enumerate(self.querys):
            for q in query:
                img_fns = []
                for img_idx in q:
                    img_fns.append(img_idx if img_idx >= 0 else None)
                num_zero_history = (q < 0).sum()
                num_zero_history_list.append(int(num_zero_history))
                values.append(
                    dict(
                        num_zero_history=num_zero_history,
                        episode_index=i,
                        img_fns=img_fns,
                        query_index=q,
                        episode_length=self._episode_length[i],
                    )
                )
        return values, num_zero_history_list

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        value = self.values[idx]
        img = self.get_image(value["img_fns"], value["episode_index"]).float().permute(0, 3, 1, 2)
        down = torch.nn.Upsample(size=(300,300))
        img = down(img)
        lang = self.get_language_instruction(value["img_fns"], value["episode_index"])
        ee_pos_cmd, ee_rot_cmd, gripper_cmd, joint, tar_obj_pose = self.get_ee_data(
            value["episode_index"], value["query_index"], value["num_zero_history"]
        )
        terminate_episode = self.get_episode_status(
            value["episode_length"], value["query_index"], value["num_zero_history"]
        )
        sample_obs = {
            "image": img.float(),
            # we permute the channel dimension to the second dimension to cope with rt1's convolution layers
            "natural_language_embedding": torch.tensor(lang).float(),
            "joint_position": torch.tensor(joint).float(),
            "tar_obj_pose": torch.tensor(tar_obj_pose).float(),
        }
        sample_action = {
            "world_vector": torch.tensor(ee_pos_cmd).float(),
            "rotation_delta": torch.tensor(ee_rot_cmd).float(),
            "gripper_closedness_action": torch.tensor(gripper_cmd).float(),
            "terminate_episode": torch.tensor(terminate_episode.argmax(-1)).long(),
        }

        return sample_obs, sample_action

    def get_image(self, img_fns, episode_index):
        """
        This function generates the step for current frame and history frames
        Parameters:
        - img_fns (list of int or None): indices of frames used in a specific step
        - episode_index (int): index of episode which the step belongs to
        
        Returns:
        - keys(list of tensors): history steps for each data
        
        """
        imgs = []
        for img_fn in img_fns:
            img_multi_view = []
            for c_v in self._cam_view:
                img_multi_view.append(
                    np.array(self._episodes[episode_index][img_fn]["observation"]["image_{}".format(self._cam_dict[c_v])])
                    if img_fn != None
                    else np.zeros_like(self._episodes[episode_index][img_fns[-1]]["observation"]["image_{}".format(self._cam_dict[c_v])])
                )
            img = np.concatenate(img_multi_view, axis=0)
            imgs.append(torch.from_numpy(img[:, :, :3]))
        return torch.stack(imgs, dim=0) / 255.0

    def get_ee_data(self, episode_index, query_index, pad_step_num):
        """
        This function reads ground truth robot actions, robot joint status and target object position and orientation:
        Parameters:
        - episode_index(int): index of episode which the step belongs to
        - query_index(tensor): index where exact data is read, padded zeros has a special index of -1
        - pad_step_num(int): how many timestep of zeros is padded
        Returns:
        - ee_pos_cmd(np.array): stores the ground truth command for robot move in position(x, y, z)
        - ee_rot_cmd(np.array): stores the ground truth command for robot move in rotation(rx, ry, rz)
        - gripper_cmd(np.array): stores the ground truth command for robot's gripper open or close
        - joint(np.array): stores the robot's joint status, which can be used to calculate ee's position
        - tar_obj_pose: stores the target object's position and orientation (x, y, z, rx, ry, rz)
        """
        start_idx = query_index[(query_index > -1).nonzero()[0, 0]]
        end_idx = query_index[-1]
        # position has 3 dimensions [x, y, z]
        ee_pos_cmd = np.zeros([pad_step_num, 3])
        # rotation has 3 dimensions [rx, ry, rz]
        ee_rot_cmd = np.zeros([pad_step_num, 3])
        # gripper has 1 dimension which controls open/close of the gripper
        gripper_cmd = np.zeros([pad_step_num, 1])
        # we are using Franka Panda robot, whose has 9 dofs of joint
        joint = np.zeros([pad_step_num, 9])
        # tar_obj_pose is 7 dimension [x,y,z,rx,ry,rz,w]
        # however, in this version we are not using tar_obj_pose
        tar_obj_pose = np.zeros([pad_step_num, 7])
        
        for step in range(start_idx, end_idx+1):
            ee_pos_cmd = np.vstack(
                (
                    ee_pos_cmd,
                    self._episodes[episode_index][step]["action_km"][:3].numpy().reshape(-1,3)
                )
            )
            ee_rot_cmd = np.vstack(
                (
                    ee_rot_cmd,
                    self._episodes[episode_index][step]["action_km"][3:6].numpy().reshape(-1,3)
                )
            )
            gripper_cmd = np.vstack(
                (
                    gripper_cmd,
                    self._episodes[episode_index][step]["action_km"][-2].numpy().reshape(-1,1)
                )
            )
            joint = np.vstack(
                (
                    joint,
                    np.zeros([1,9])
                )
            )
            tar_obj_pose = np.vstack(
                (
                    tar_obj_pose,
                    np.zeros([1,7])
                )
            )
        
        # print(ee_pos_cmd.shape, ee_rot_cmd.shape, gripper_cmd.shape, joint.shape, tar_obj_pose.shape)
        
        # ee_pos_cmd = np.vstack(
        #     (
        #         ee_pos_cmd,
        #         raw_data.loc[
        #             start_idx:end_idx,
        #             [f"kinematic_delta_ee_position_{ax}" for ax in ["x", "y", "z"]],
        #         ].to_numpy(),
        #     )
        # )
        # ee_rot_cmd = np.vstack(
        #     (
        #         ee_rot_cmd,
        #         raw_data.loc[
        #             start_idx:end_idx,
        #             [f"kinematic_delta_ee_rotation_{ax}" for ax in ["x", "y", "z"]],
        #         ].to_numpy(),
        #     )
        # )
        # joint = np.vstack(
        #     (
        #         joint,
        #         raw_raw_data.loc[
        #             start_idx:end_idx,
        #             [f"joint_{str(ax)}" for ax in range(self._robot_dof)],
        #         ].to_numpy(),
        #     )
        # )
        # tar_obj_pose = np.vstack(
        #     (
        #         tar_obj_pose,
        #         raw_raw_data.loc[
        #             start_idx:end_idx,
        #             [
        #                 f"tar_obj_pose_{ax}"
        #                 for ax in ["x", "y", "z", "rx", "ry", "rz", "rw"]
        #             ],
        #         ].to_numpy(),
        #     )
        # )
        # gripper_data = (
        #     raw_data.loc[start_idx:end_idx, "action_gripper"]
        #     .to_numpy()
        #     .reshape(-1, 1)
        # )
        # gripper_cmd = np.vstack((gripper_cmd, gripper_data))
        return ee_pos_cmd, ee_rot_cmd, gripper_cmd, joint, tar_obj_pose

    def get_language_instruction(self, img_fns, episode_index):
        language_embedding = np.zeros([self._time_sequence_length, self._language_embedding_size])
        for i, img_fn in enumerate(img_fns):
            if img_fn != None:
                language_embedding[i] = self._episodes[episode_index][img_fn]["language_embedding"].numpy()
        
        return language_embedding

    def get_episode_status(self, episode_length, query_index, pad_step_num):
        """
        This function is to find whether current frame and history frame is start or middle or end of the episode:
        Parameters:
        - episode_length(int): length of current episode
        - query_index(tensor): index where exact data is read, padded zeros has a special index of -1
        - pad_step_num(int): how many timestep of zeros is padded
        Returns:
        - episode_status(np.array): specifies status(start, middle or end) of each frame in history
        """
        start_idx = query_index[(query_index > -1).nonzero()[0, 0]]
        end_idx = query_index[-1]
        episode_status = np.zeros([pad_step_num, 4], dtype=np.int32)
        episode_status[:, -1] = 1
        for i in range(start_idx, end_idx + 1):
            status = np.array(
                [i == 0, i not in [0, episode_length - 2], i == episode_length - 2, 0],
                dtype=np.int32,
            )
            episode_status = np.vstack((episode_status, status))
        if pad_step_num > 0:
            episode_status[pad_step_num] = np.array([1, 0, 0, 0])
        return episode_status


def load_config_from_json(json_path):
    with open(json_path, "r") as f:
        config = json.load(f)
    return config


if __name__ == "__main__":
    args = load_config_from_json("train_config.json")
    dataset, _ = build_dataset(
        data_path=args["data_path"],
        time_sequence_length=args["time_sequence_length"],
        predicting_next_ts=args["predicting_next_ts"],
        num_train_episode=args["num_train_episode"],
        num_val_episode=args["num_val_episode"],
        cam_view=args["cam_view"],
        language_embedding_size=args["network_configs"]["language_embedding_size"],
    )
    # dataset = dataset[:100]

    wv_x = []
    wv_y = []
    wv_z = []
    rd_x = []
    rd_y = []
    rd_z = []
    from maruya24_rt1.tokenizers import action_tokenizer
    from gym import spaces
    from collections import OrderedDict
    import matplotlib.pyplot as plt

    output_tensor_space = spaces.Dict(
        OrderedDict(
            [
                ("terminate_episode", spaces.Discrete(4)),
                (
                    "world_vector",
                    spaces.Box(low=-0.025, high=0.025, shape=(3,), dtype=np.float32),
                ),
                (
                    "rotation_delta",
                    spaces.Box(
                        low=-np.pi / 20,
                        high=np.pi / 20,
                        shape=(3,),
                        dtype=np.float32,
                    ),
                ),
                (
                    "gripper_closedness_action",
                    spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                ),
            ]
        )
    )
    at = action_tokenizer.RT1ActionTokenizer(
        output_tensor_space, vocab_size=256  # action space
    )
    dataloader = DataLoader(dataset, batch_size=64, num_workers=64)
    for i, batch in tqdm(enumerate(dataloader), total=len(dataset) // 64):
        batch = at.tokenize(batch[1])
        for i in range(batch.size(0)):
            wv_x.append(int(batch[i, -1, 1]))
            wv_y.append(int(batch[i, -1, 2]))
            wv_z.append(int(batch[i, -1, 3]))
            rd_x.append(int(batch[i, -1, 4]))
            rd_y.append(int(batch[i, -1, 5]))
            rd_z.append(int(batch[i, -1, 6]))
        # print(batch)
    plt.subplot(2, 3, 1)
    plt.title("world_vector_x")
    plt.hist(wv_x, bins=256, range=(0, 256))
    plt.xlim(0, 256)
    plt.subplot(2, 3, 2)
    plt.title("world_vector_y")
    plt.hist(wv_y, bins=256, range=(0, 256))
    plt.xlim(0, 256)
    plt.subplot(2, 3, 3)
    plt.title("world_vector_z")
    plt.hist(wv_z, bins=256, range=(0, 256))
    plt.xlim(0, 256)
    plt.subplot(2, 3, 4)
    plt.title("rotation_delta_x")
    plt.hist(rd_x, bins=256, range=(0, 256))
    plt.xlim(0, 256)
    plt.subplot(2, 3, 5)
    plt.title("rotation_delta_y")
    plt.hist(rd_y, bins=256, range=(0, 256))
    plt.xlim(0, 256)
    plt.subplot(2, 3, 6)
    plt.title("rotation_delta_z")
    plt.hist(rd_z, bins=256, range=(0, 256))
    plt.xlim(0, 256)
    plt.show()
