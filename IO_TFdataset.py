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

# This loads the tfrecord dataset iteratively without the need to prefetch the whole dataset.

def create_tf_dataset(tfrecord_directory, train_val_split):
    dataset_builder = tfds.builder_from_directory(tfrecord_directory)
    tf_train_dataset, tf_val_dataset = dataset_builder.as_dataset(split=["train[:{}%]".format(train_val_split), "train[{}%:]".format(train_val_split)])
    return tf_train_dataset, tf_val_dataset

def build_dataset(
    data_path,
    time_sequence_length=6,
    train_val_split=95,
    cam_view=["front"],
    language_embedding_size=512,
):
    """
    This function is for building the training and validation dataset

    Parameters:
    - data_path(str): locates the path where the dataset is stored
            the dataset path should have the following file structures:
                - [robotname]_[taskname]
                    - [cam_view_0]
                        - data_000
                            - rgb # where those image stored
                                - image_001.png
                                - image_002.png
                                - ...
                            - results.csv # robot actions stored
                            - results_raw.csv # joint and target object position stored
                        - data_001
                        - ...
                    - [cam_view_1]
                        - data_000
                        - data_001
                        - ...
                    - ...
    - time_sequence_length(int) : number of history length input for RT-1 model,
        6 means current frame image and past 5 frames of images will be packed and input to RT-1
    - predicting_next_ts(bool) : in our dataset's results.csv and results_raw.csv, we stored current frame's action and joint status.
        if we want to predict next frame's action, this option needs to be True and result in the 1 step offset reading on csv files
        this differs between the samplings method of different dataset.
    - num_train_episode(int) : specifies numbers of training episodes
    - num_train_episode(int) : specifies numbers of validation episodes
    - cam_view(list of strs) : camera views used for training.

    Returns:
    - train_dataset(torch.utils.data.Dataset)
    - val_dataset(torch.utils.data.Dataset)
    """

    cam_dict = {
        "front": 0,
        "fronttop": 1,
        "root": 2,
        "root_shoulder": 3,
        "side": 4,
        "topdown": 5,
        "wrist": 6,
    }

    tfrecord_directory = data_path
    tf_train_dataset, tf_val_dataset = create_tf_dataset(tfrecord_directory, train_val_split)
    
    train_indices = []
    train_num_episodes = tf_train_dataset.cardinality().numpy()
    for i, eps in enumerate(tf_train_dataset.take(train_num_episodes)):
        num_steps = eps["steps"].cardinality().numpy()
        for j, _ in enumerate(eps["steps"].take(num_steps)):
            train_indices.append([i, j])
    
    val_indices = []
    val_num_episodes = tf_val_dataset.cardinality().numpy()
    for i, eps in enumerate(tf_val_dataset.take(val_num_episodes)):
        num_steps = eps["steps"].cardinality().numpy()
        for j, _ in enumerate(eps["steps"].take(num_steps)):
            val_indices.append([i, j])
    
    train_dataset = TFRecordDataset(
        dataset=tf_train_dataset,
        indices=train_indices,
        time_sequence_length=time_sequence_length,
        cam_view=cam_view,
        cam_dict=cam_dict,
        language_embedding_size=language_embedding_size,
    )
    
    val_dataset = TFRecordDataset(
        dataset=tf_val_dataset,
        indices=val_indices,
        time_sequence_length=time_sequence_length,
        cam_view=cam_view,
        cam_dict=cam_dict,
        language_embedding_size=language_embedding_size,
    )

    
    return train_dataset, val_dataset


class TFRecordDataset(Dataset):
    def __init__(
        self,
        dataset,
        indices,
        time_sequence_length=6,
        cam_view=["front"],
        cam_dict=None,
        language_embedding_size=512,
    ):
        self._dataset = dataset
        self._indices = indices
        self._cam_view = cam_view
        self._cam_dict = cam_dict
        self._time_sequence_length = time_sequence_length
        self._language_embedding_size = language_embedding_size

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        
        self._eps_step = self._indices[idx]
        for eps in self._dataset.skip(self._eps_step[0]).take(1):
            self._episode = [step for step in eps["steps"]]
        
        img = self.get_image().float().permute(0, 3, 1, 2)
        down = torch.nn.Upsample(size=(300,300))
        img = down(img)
        lang = self.get_language_instruction()
        ee_pos_cmd, ee_rot_cmd, gripper_cmd, joint, tar_obj_pose = self.get_ee_data()
        terminate_episode = self.get_episode_status()
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

    def get_image(self):
        """
        This function generates the step for current frame and history frames
        Parameters:
        - img_fns (list of int or None): indices of frames used in a specific step
        - episode_index (int): index of episode which the step belongs to
        
        Returns:
        - keys(list of tensors): history steps for each data
        
        """
        imgs = []
        step = self._eps_step[1]
        for img_fn in range(step-self._time_sequence_length+1, step+1):
            img_multi_view = []
            for c_v in self._cam_view:
                img_multi_view.append(
                    np.array(self._episode[img_fn]["observation"]["image_{}".format(self._cam_dict[c_v])])
                    if img_fn >= 0
                    else np.zeros_like(self._episode[step]["observation"]["image_{}".format(self._cam_dict[c_v])])
                )
            img = np.concatenate(img_multi_view, axis=0)
            imgs.append(torch.from_numpy(img[:, :, :3]))
        return torch.stack(imgs, dim=0) / 255.0
    
    def get_language_instruction(self):
        """
        since we are only training single-task model, this language embedding is set as constant.
        modify it to language instructions if multi-task model is training.
        it seems that google directly loads embedded language instruction from its language model
        this results in our loading a language embedding instead of language sentence
        """
        language_embedding = np.zeros([self._time_sequence_length, self._language_embedding_size])
        step = self._eps_step[1]
        for i, img_fn in enumerate(range(step-self._time_sequence_length+1, step+1)):
            if img_fn >= 0:
                language_embedding[i] = self._episode[img_fn]["language_embedding"].numpy()
        
        return language_embedding

    def get_ee_data(self):
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
        step = self._eps_step[1]
        # position has 3 dimensions [x, y, z]
        ee_pos_cmd = np.zeros([self._time_sequence_length, 3])
        # rotation has 3 dimensions [rx, ry, rz]
        ee_rot_cmd = np.zeros([self._time_sequence_length, 3])
        # gripper has 1 dimension which controls open/close of the gripper
        gripper_cmd = np.zeros([self._time_sequence_length, 1])
        # we are using Franka Panda robot, whose has 9 dofs of joint
        joint = np.zeros([self._time_sequence_length, 9])
        # tar_obj_pose is 7 dimension [x,y,z,rx,ry,rz,w]
        # however, in this version we are not using tar_obj_pose
        tar_obj_pose = np.zeros([self._time_sequence_length, 7])
        
        for i, s in enumerate(range(step-self._time_sequence_length+1, step+1)):
            if s >= 0:
                ee_pos_cmd[i] = self._episode[s]["action_km"][:3].numpy().reshape(-1)
                ee_rot_cmd[i] = self._episode[s]["action_km"][3: 6].numpy().reshape(-1)
                gripper_cmd[i] = self._episode[s]["action_km"][-2].numpy().reshape(-1)
        # print(ee_pos_cmd.shape, ee_rot_cmd.shape, gripper_cmd.shape, joint.shape, tar_obj_pose.shape)
        
        return ee_pos_cmd, ee_rot_cmd, gripper_cmd, joint, tar_obj_pose


    def get_episode_status(self):
        """
        This function is to find whether current frame and history frame is start or middle or end of the episode:
        Parameters:
        - episode_length(int): length of current episode
        - query_index(tensor): index where exact data is read, padded zeros has a special index of -1
        - pad_step_num(int): how many timestep of zeros is padded
        Returns:
        - episode_status(np.array): specifies status(start, middle or end) of each frame in history
        """
        step = self._eps_step[1]
        start_idx = step - self._time_sequence_length + 1 if step >= self._time_sequence_length - 1 else 0
        end_idx = step
        pad_step_num = self._time_sequence_length - (end_idx - start_idx + 1)
        episode_status = np.zeros([pad_step_num, 4], dtype=np.int32)
        episode_status[:, -1] = 1
        for i in range(start_idx, end_idx + 1):
            status = np.array(
                [i == 0, i not in [0, len(self._episode) - 1], i == len(self._episode) - 1, 0],
                dtype=np.int32,
            )
            episode_status = np.vstack((episode_status, status))
        # if pad_step_num > 0:
        #     episode_status[pad_step_num] = np.array([1, 0, 0, 0])
        return episode_status


def load_config_from_json(json_path):
    with open(json_path, "r") as f:
        config = json.load(f)
    return config
