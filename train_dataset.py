import random
import pickle

import logging
import torch
import cv2
import os

from torch.utils.data.dataset import Dataset
import numpy as np
import cvbase
from .util.STTN_mask import create_random_shape_with_random_motion
import imageio
from .util.flow_utils import region_fill as rf

logger = logging.getLogger('base')


class VideoBasedDataset(Dataset):
    def __init__(self, opt, dataInfo):
        self.opt = opt
        self.sampleMethod = opt['sample']
        self.dataInfo = dataInfo
        self.height, self.width = self.opt['input_resolution']
        self.frame_path = dataInfo['frame_path']
        self.frame_ego_path=dataInfo['frame_ego_path']
        self.flow_path = dataInfo['flow_path']  # The path of the optical flows
        self.train_list = os.listdir(self.frame_path)
        self.video_len=dataInfo['video_len']
        self.sequenceLen = self.opt['num_frames']
        self.num_ref = self.opt['num_ref']
        self.ref_length = self.opt['ref_length']

        self.flow2rgb = opt['flow2rgb']  # whether to change flow to rgb domain
        self.flow_direction = opt[
            'flow_direction']  # The direction must be in ['for', 'back', 'bi'], indicating forward, backward and bidirectional flows

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):
        try:
            item = self.load_item(idx)
        except:
            print('Loading error: ' + self.train_list[idx])
            item = self.load_item(0)
        return item

    def get_ref_index(self, f, neighbor_ids, length, ref_length, num_ref):
        ref_index = []
        if num_ref == -1:
            for i in range(0, length, ref_length):
                if i not in neighbor_ids:
                    ref_index.append(i)
        else:
            start_idx = max(0, f - ref_length * (num_ref // 2))
            end_idx = min(length-1, f + ref_length * (num_ref // 2))
            for i in range(start_idx, end_idx + 1, ref_length):
                if i not in neighbor_ids:
                    if len(ref_index) > num_ref:
                        break
                    ref_index.append(i)
            # Filling to satisfy the number of num_ref
            if len(ref_index)<num_ref and f - ref_length * (num_ref // 2)<0:
                max_ref_index=max(ref_index)
                for i in range(num_ref-len(ref_index)):
                    ref_index.append(max_ref_index+(i+1)*ref_length)
            else:
                min_ref_index=min(ref_index)
                for i in range(num_ref-len(ref_index)):
                    ref_index.insert(0,min_ref_index-(i+1)*ref_length)
        return ref_index

    def frameSample(self, frameLen, sequenceLen):
        if self.sampleMethod == 'random':
            indices = [i for i in range(frameLen)]
            sampleIndices = random.sample(indices, sequenceLen)
        elif self.sampleMethod == 'random_window':
            middle_idx = random.randint(self.opt['sample_window_size']//2, frameLen-self.opt['sample_window_size']//2)
            indices = [i for i in range(middle_idx-self.opt['sample_window_size']//2,middle_idx+self.opt['sample_window_size']//2)]
            sampleIndices = random.sample(indices, sequenceLen)
        elif self.sampleMethod == 'seq_interval': #sequence 'sequenceLen' frames with interval of 'seq_interval_frames'
            pivot = random.randint(0, frameLen-sequenceLen*self.opt['seq_interval_frames'] - 1)
            sampleIndices = [i for i in range(pivot, pivot+sequenceLen*self.opt['seq_interval_frames'], self.opt['seq_interval_frames'])]
        elif self.sampleMethod == 'seqframes_with_refframes':
            ref_length = self.opt['ref_length']
            num_ref = self.opt['num_ref']
            neighbor_stride = self.opt['num_frames']//2
            video_length=frameLen

            f = random.randint(0, video_length-neighbor_stride)
            if f - neighbor_stride<0:
                neighbor_ids = [i for i in range(0,2*neighbor_stride+1)]
            elif f + neighbor_stride>=video_length:
                neighbor_ids = [i for i in range(video_length-2*neighbor_stride-1,video_length)]
            else:
                neighbor_ids = [i for i in range(f - neighbor_stride, f + neighbor_stride + 1)]
            
            ref_ids = self.get_ref_index(f, neighbor_ids, video_length, ref_length, num_ref)
            sampleIndices= neighbor_ids+ref_ids
        else:
            raise ValueError('Cannot determine the sample method {}'.format(self.sampleMethod))
        return sampleIndices

    def load_item(self, idx):
        video = self.train_list[idx]
        frame_dir = os.path.join(self.frame_path, video)
        frame_ego_dir = os.path.join(self.frame_ego_path, video)
        forward_flow_dir = os.path.join(self.flow_path, video, 'forward_flo')
        backward_flow_dir = os.path.join(self.flow_path, video, 'backward_flo')
        frameLen = self.video_len#len(os.listdir(frame_dir)) # TODO original: frameLen = self.name2length[video]
        flowLen = frameLen - 1
        assert frameLen > self.sequenceLen, 'Frame length {} is less than sequence length'.format(frameLen)
        sampledIndices = self.frameSample(frameLen, self.sequenceLen)

        frames, frames_ego, forward_flows, backward_flows = [], [], [], []
        for i in range(len(sampledIndices)):
            frame = self.read_frame(os.path.join(frame_dir, '{:05d}.jpg'.format(sampledIndices[i])), self.height,
                                    self.width)
            frame_ego = self.read_frame(os.path.join(frame_ego_dir, '{:05d}.jpg'.format(sampledIndices[i])), self.height,
                                    self.width)
            frames.append(frame)
            frames_ego.append(frame_ego)
            if self.flow_direction == 'for':
                forward_flow = self.read_forward_flow(forward_flow_dir, sampledIndices[i], flowLen)
                forward_flows.append(forward_flow)
            elif self.flow_direction == 'back':
                backward_flow = self.read_backward_flow(backward_flow_dir, sampledIndices[i])
                backward_flows.append(backward_flow)
            elif self.flow_direction == 'bi':
                forward_flow = self.read_forward_flow(forward_flow_dir, sampledIndices[i], flowLen)
                forward_flows.append(forward_flow)
                backward_flow = self.read_backward_flow(backward_flow_dir, sampledIndices[i])
                backward_flows.append(backward_flow)
            else:
                raise ValueError('Unknown flow direction mode: {}'.format(self.flow_direction))
        inputs = {'frames': frames, 'frames_ego': frames_ego, 'forward_flo': forward_flows, 'backward_flo': backward_flows}
        inputs = self.to_tensor(inputs)
        inputs['frames'] = (inputs['frames'] / 255.) * 2 - 1
        inputs['frames_ego'] = (inputs['frames_ego'] / 255.) * 2 - 1
        return inputs


    def read_frame(self, path, height, width):
        frame = imageio.imread(path)
        frame = cv2.resize(frame, (width, height), cv2.INTER_LINEAR)
        return frame

    def read_forward_flow(self, forward_flow_dir, sampledIndex, flowLen):
        if sampledIndex >= flowLen:
            sampledIndex = flowLen - 1
        flow = cvbase.read_flow(os.path.join(forward_flow_dir, '{:05d}.flo'.format(sampledIndex)))
        height, width = flow.shape[:2]
        flow = cv2.resize(flow, (self.width, self.height), cv2.INTER_LINEAR)
        flow[:, :, 0] = flow[:, :, 0] / width * self.width
        flow[:, :, 1] = flow[:, :, 1] / height * self.height
        return flow

    def read_backward_flow(self, backward_flow_dir, sampledIndex):
        if sampledIndex == 0:
            sampledIndex = 0
        else:
            sampledIndex -= 1
        flow = cvbase.read_flow(os.path.join(backward_flow_dir, '{:05d}.flo'.format(sampledIndex)))
        height, width = flow.shape[:2]
        flow = cv2.resize(flow, (self.width, self.height), cv2.INTER_LINEAR)
        flow[:, :, 0] = flow[:, :, 0] / width * self.width
        flow[:, :, 1] = flow[:, :, 1] / height * self.height
        return flow

    def to_tensor(self, data_list):
        """

        Args:
            data_list: A list contains multiple numpy arrays

        Returns: The stacked tensor list

        """
        keys = list(data_list.keys())
        for key in keys:
            if data_list[key] is None or data_list[key] == []:
                data_list.pop(key)
            else:
                item = data_list[key]
                if not isinstance(item, list):
                    item = torch.from_numpy(np.transpose(item, (2, 0, 1))).float()  # [c, h, w]
                else:
                    item = np.stack(item, axis=0)
                    if len(item.shape) == 3:  # [t, h, w]
                        item = item[:, :, :, np.newaxis]
                    item = torch.from_numpy(np.transpose(item, (0, 3, 1, 2))).float()  # [t, c, h, w]
                data_list[key] = item
        return data_list

