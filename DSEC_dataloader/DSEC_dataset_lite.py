import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from DSEC_dataloader.data_augmentation import Compose,CenterCrop,RandomCrop,RandomRotationFlip,Random_event_drop,downsample_data
import torch.nn.functional as F
from configs.parser import YAMLParser
import h5py
# from DSEC_dataloader.event_representations import EventSlicer,rectify_events, cumulate_spikes_into_frames,VoxelGrid
import tqdm
from os import walk



def binary_search_array(array, x, left=None, right=None, side="left"):
    """
    Binary search through a sorted array.
    """

    left = 0 if left is None else left
    right = len(array) - 1 if right is None else right
    mid = left + (right - left) // 2

    if left > right:
        return left if side == "left" else right

    if array[mid] == x:
        return mid

    if x < array[mid]:
        return binary_search_array(array, x, left=left, right=mid - 1)

    return binary_search_array(array, x, left=mid + 1, right=right)

class DSECDatasetLite(Dataset):
    def __init__(self, config, file_list: str, stereo=False, transform=None, scale_factor: float = 1):
        self.config = config
        self.flow_path = os.path.join(self.config['data']['path'], 'gt_tensors')
        self.mask_path = os.path.join(self.config['data']['path'], 'mask_tensors')
        self.input = self.config['model']['encoding']
        self.num_frames_per_ts = config['data']['num_frames']
        self.num_chunks =  config['data']['num_chunks']
        self.scale_factor = scale_factor
        self.height = int(self.config['loader']['resolution'][0])
        self.width = int(self.config['loader']['resolution'][1])
        self.num_bins = self.num_frames_per_ts * self.num_chunks
        self.new_sequence = True
        if not config['data']['preprocessed']:
            self.events_path = os.path.join(self.config['data']['path'], 'event_tensors', '01lists','left')
            # if (self.input == "voxel"):
            #     self.voxel_grid = VoxelGrid((self.num_bins, self.height, self.width))
        else:
            if (self.input == "voxel"):
                if self.config['loader']['polarity']:
                    self.events_path = os.path.join(self.config['data']['path'], 'event_tensors', '{}bins'.format(str(self.num_frames_per_ts).zfill(2)),'left')
                else:
                    self.events_path = os.path.join(self.config['data']['path'], 'event_tensors',
                                                    '{}bins_pol'.format(str(self.num_frames_per_ts).zfill(2)), 'left')


            elif (self.input == "cnt"):
                self.events_path = os.path.join(self.config['data']['path'], 'event_tensors', '{}frames'.format(str(self.num_frames_per_ts).zfill(2)),'left')


        if self.num_chunks == 2:
            file_list = file_list+"_split_doubleseq.csv"
            sequence_file = os.path.join(self.config['data']['path'], 'sequence_lists', file_list)

        elif self.num_chunks == 1:
            # self.files= []
            # for (dirpath, dirnames, filenames) in walk(self.events_path):
            #     self.files.extend(filenames)
            #     break
            file_list = file_list+"_split_seq.csv"
            sequence_file = os.path.join(self.config['data']['path'], 'sequence_lists', file_list)
        self.files = pd.read_csv(sequence_file, header=None)
        self.transform = transform

    def __len__(self):
        return len(self.files)



    def get_events_idx(self, events_ts, t_start,t_end):
        """
        Find closest event index for a given timestamp through binary search.
        """
        event_id0 = binary_search_array(events_ts, t_start,side='left')
        event_id1 = binary_search_array(events_ts, t_end,side='left')

        return event_id0, event_id1



    def __getitem__(self, idx):

        # if target_file_1.split('_')[-1] == "0001":
        #     self.new_sequence = False
        if self.num_chunks ==1:
            # target_file_1 = self.files[idx]
            target_file_1 = self.files.iloc[idx, 0]
            mask = torch.from_numpy(np.load(os.path.join(self.mask_path, target_file_1)))
            label = torch.from_numpy(np.load(os.path.join(self.flow_path, target_file_1)))
            seq_folder1 = "_".join(target_file_1.split('_')[:-1])
        elif self.num_chunks ==2:

            target_file_1 = self.files.iloc[idx, 0]
            target_file_2 = self.files.iloc[idx, 1]
            mask = torch.from_numpy(np.load(os.path.join(self.mask_path, target_file_2)))
            label = torch.from_numpy(np.load(os.path.join(self.flow_path, target_file_2)))
            seq_folder1 = "_".join(target_file_1.split('_')[:-1])
            seq_folder2= "_".join(target_file_2.split('_')[:-1])

        if self.config['data']['preprocessed']:

            # load preprocessed event cnt images/voxel representation
            chunk = torch.from_numpy(np.load(os.path.join(self.events_path , seq_folder1, target_file_1), allow_pickle=True))
            if self.num_chunks ==2:
                eventsL2 = torch.from_numpy(np.load(os.path.join(self.events_path, seq_folder2, target_file_2), allow_pickle=True))
                chunk = torch.cat((chunk, eventsL2), axis=0)
            elif self.num_chunks > 2:
                raise  AttributeError

        else:
            chunk = {}
            events = np.load(os.path.join(self.events_path, target_file_1),allow_pickle=True) #dict p x y ts
            chunk["ts"] = torch.from_numpy(events[0]["t"])
            chunk["x"] = torch.from_numpy(events[0]["x"])
            chunk["y"] = torch.from_numpy(events[0]["y"])
            chunk["p"] = torch.from_numpy(events[0]["p"])




        return  chunk, mask, label




if __name__ == "__main__":
    config = YAMLParser("../configs/train_DSEC_supervised_Spikingformer.yml").config
    config = YAMLParser.combine_entries(config)
    config['data']['path'] = '../data/Datasets/DSEC/saved_flow_data'
    config['data']['preprocessed'] = False
    config['model']['encoding'] = "list"

    transform_train = Compose([
        RandomRotationFlip((0,0),config['loader']['augment_prob'][0],config['loader']['augment_prob'][1]),
        RandomCrop(config['loader']['crop'][0],config['loader']['crop'][1]),
        # Random_event_drop()
    ])
    transform_valid = Compose([
        CenterCrop(config['loader']['resolution'][0],config['loader']['resolution'][0]),
    ])
    # Create training dataset
    print("Training Dataset ...")
    train_dataset = DSECDatasetLite(
        config,
        file_list='train',
        stereo=False,
        transform=None,
        scale_factor=1
    )

    for chunk, mask, label in train_dataset:
        print(torch.max(chunk))
        print(torch.min(chunk))
        pass




