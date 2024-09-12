import numpy
import random
import math
from PIL import Image

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
from torchvision.transforms import ColorJitter
import torch.nn.functional as F

import pandas

import pdb

def get_compressed_events(event_path):
    try:
        events_dict = numpy.load(event_path)
        events_x = events_dict["x"]
        events_y = events_dict["y"]
        events_t = events_dict["t"]
        events_p = events_dict["p"]

        events_p_ = 2*events_p - 1
        events_dtype = numpy.stack([events_t*1e-9, events_x, events_y, events_p_], axis=1).astype(numpy.float64)
        
        return events_dtype

    except OSError:
        print("No file " + event_path)
        print("Creating an array of zeros!")
        return 0

def get_events(event_path):
    try:
        f = pandas.read_hdf(event_path, "myDataset")
        return f[['ts', 'x', 'y', 'p']]

    except OSError:
        print("No file " + event_path)
        print("Creating an array of zeros!")
        return 0

def read_flo(flow_path):
    with open(flow_path, 'rb') as f:
        magic = numpy.fromfile(f, numpy.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
        else:
            w = numpy.fromfile(f, numpy.int32, count=1)
            h = numpy.fromfile(f, numpy.int32, count=1)
            data = numpy.fromfile(f, numpy.float32, count=int(2 * w * h))
            # Reshape data into 3D array (columns, rows, bands)
            data2D = numpy.resize(data, (h[0], w[0], 2))
            return data2D


"""Propagates x_indices and y_indices by their flow, as defined in x_flow, y_flow. x_mask and y_mask are zeroed out at each pixel where the indices leave the image.
The optional scale_factor will scale the final displacement."""
def prop_flow(x_flow, y_flow, x_indices, y_indices, x_mask, y_mask, scale_factor=1.0):
    flow_x_interp = cv2.remap(x_flow, x_indices, y_indices, cv2.INTER_NEAREST)
    flow_y_interp = cv2.remap(y_flow, x_indices, y_indices, cv2.INTER_NEAREST)

    x_mask[flow_x_interp == 0] = False
    y_mask[flow_y_interp == 0] = False

    x_indices += flow_x_interp * scale_factor
    y_indices += flow_y_interp * scale_factor
    return

"""The ground truth flow maps are not time synchronized with the grayscale images. Therefore, we need to propagate the ground truth flow over the time between two images.
This function assumes that the ground truth flow is in terms of pixel displacement, not velocity. Pseudo code for this process is as follows:
x_orig = range(cols)      y_orig = range(rows)
x_prop = x_orig           y_prop = y_orig
Find all GT flows that fit in [image_timestamp, image_timestamp+image_dt].
for all of these flows:
  x_prop = x_prop + gt_flow_x(x_prop, y_prop)
  y_prop = y_prop + gt_flow_y(x_prop, y_prop)
The final flow, then, is x_prop - x-orig, y_prop - y_orig.
Note that this is flow in terms of pixel displacement, with units of pixels, not pixel velocity.
Inputs:
  x_flow_in, y_flow_in - list of numpy arrays, each array corresponds to per pixel flow at each timestamp.
  gt_timestamps - timestamp for each flow array.  start_time, end_time - gt flow will be estimated between start_time and end time."""
def estimate_corresponding_gt_flow(x_flow_in, y_flow_in, gt_timestamps, start_time, end_time):
    x_flow_in = numpy.array(x_flow_in, dtype=numpy.float64)
    y_flow_in = numpy.array(y_flow_in, dtype=numpy.float64)
    gt_timestamps = numpy.array(gt_timestamps, dtype=numpy.float64)
    start_time = numpy.array(start_time, dtype=numpy.float64)
    end_time = numpy.array(end_time, dtype=numpy.float64)

    # Each gt flow at timestamp gt_timestamps[gt_iter] represents the displacement between gt_iter and gt_iter+1.
    gt_iter = numpy.searchsorted(gt_timestamps, start_time, side='right') - 1
    gt_dt = gt_timestamps[gt_iter + 1] - gt_timestamps[gt_iter]
    x_flow = numpy.squeeze(x_flow_in[gt_iter, ...])
    y_flow = numpy.squeeze(y_flow_in[gt_iter, ...])

    dt = end_time - start_time

    # No need to propagate if the desired dt is shorter than the time between gt timestamps.
    if gt_dt > dt:
        return x_flow*dt/gt_dt, y_flow*dt/gt_dt
    # pdb.set_trace()
    
    x_indices, y_indices = numpy.meshgrid(numpy.arange(x_flow.shape[1]), numpy.arange(x_flow.shape[0]))
    x_indices = x_indices.astype(numpy.float32)
    y_indices = y_indices.astype(numpy.float32)

    orig_x_indices = numpy.copy(x_indices)
    orig_y_indices = numpy.copy(y_indices)

    # Mask keeps track of the points that leave the image, and zeros out the flow afterwards.
    x_mask = numpy.ones(x_indices.shape, dtype=bool)
    y_mask = numpy.ones(y_indices.shape, dtype=bool)

    scale_factor = (gt_timestamps[gt_iter + 1] - start_time) / gt_dt
    total_dt = gt_timestamps[gt_iter + 1] - start_time
    
    prop_flow(x_flow, y_flow, x_indices, y_indices, x_mask, y_mask, scale_factor=scale_factor)
    gt_iter += 1

    while gt_timestamps[gt_iter + 1] < end_time:
        x_flow = numpy.squeeze(x_flow_in[gt_iter, ...])
        y_flow = numpy.squeeze(y_flow_in[gt_iter, ...])

        prop_flow(x_flow, y_flow, x_indices, y_indices, x_mask, y_mask)
        total_dt += gt_timestamps[gt_iter + 1] - gt_timestamps[gt_iter]

        gt_iter += 1
#         if gt_iter+1 >= len(gt_timestamps):
#            gt_iter -= 1
#            break


    final_dt = end_time - gt_timestamps[gt_iter]
    total_dt += final_dt

    final_gt_dt = gt_timestamps[gt_iter + 1] - gt_timestamps[gt_iter]

    x_flow = numpy.squeeze(x_flow_in[gt_iter, ...])
    y_flow = numpy.squeeze(y_flow_in[gt_iter, ...])

    scale_factor = final_dt / final_gt_dt

    prop_flow(x_flow, y_flow, x_indices, y_indices, x_mask, y_mask, scale_factor)

    x_shift = x_indices - orig_x_indices
    y_shift = y_indices - orig_y_indices
    x_shift[~x_mask] = 0
    y_shift[~y_mask] = 0

    return x_shift, y_shift
    

class FlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5
        
    def eraser_transform(self, img1, img2):
        channels = img1.shape[-1]
        ht, wd = img1.shape[:2]
        if numpy.random.rand() < self.eraser_aug_prob:
            mean_color = numpy.mean(img2.reshape(-1, channels), axis=0)
            for _ in range(numpy.random.randint(1, channels)):
                x0 = numpy.random.randint(0, wd)
                y0 = numpy.random.randint(0, ht)
                dx = numpy.random.randint(50, 100)
                dy = numpy.random.randint(50, 100)
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2

    def spatial_transform(self, img1, img2, flow):
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = numpy.maximum(
            (self.crop_size[0] + 8) / float(ht), 
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** numpy.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if numpy.random.rand() < self.stretch_prob:
            scale_x *= 2 ** numpy.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** numpy.random.uniform(-self.max_stretch, self.max_stretch)
        
        scale_x = numpy.clip(scale_x, min_scale, None)
        scale_y = numpy.clip(scale_y, min_scale, None)

        if numpy.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = flow * [scale_x, scale_y]

        if self.do_flip:
            if numpy.random.rand() < self.h_flip_prob: # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if numpy.random.rand() < self.v_flip_prob: # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        y0 = numpy.random.randint(0, img1.shape[0] - self.crop_size[0])
        x0 = numpy.random.randint(0, img1.shape[1] - self.crop_size[1])
        
        img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        return img1, img2, flow

    def __call__(self, img1, img2, flow):
        # img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow= self.spatial_transform(img1, img2, flow)

        img1 = numpy.ascontiguousarray(img1)
        img2 = numpy.ascontiguousarray(img2)
        flow = numpy.ascontiguousarray(flow)

        return img1, img2, flow

class DenseSparseAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5
        
    def eraser_transform(self, img1, img2, dimg1, dimg2):
        channels = img1.shape[-1]
        ht, wd = img1.shape[:2]
        if numpy.random.rand() < self.eraser_aug_prob:
            mean_color = numpy.mean(img2.reshape(-1, channels), axis=0)
            for _ in range(numpy.random.randint(1, channels)):
                x0 = numpy.random.randint(0, wd)
                y0 = numpy.random.randint(0, ht)
                dx = numpy.random.randint(50, 100)
                dy = numpy.random.randint(50, 100)
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color
                dimg2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2, dimg1, dimg2

    def spatial_transform(self, img1, img2, dimg1, dimg2, flow):
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = numpy.maximum(
            (self.crop_size[0] + 8) / float(ht),
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** numpy.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if numpy.random.rand() < self.stretch_prob:
            scale_x *= 2 ** numpy.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** numpy.random.uniform(-self.max_stretch, self.max_stretch)
        
        scale_x = numpy.clip(scale_x, min_scale, None)
        scale_y = numpy.clip(scale_y, min_scale, None)

        if numpy.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            dimg1 = cv2.resize(dimg1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            dimg2 = cv2.resize(dimg2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = flow * [scale_x, scale_y]

        if self.do_flip:
            if numpy.random.rand() < self.h_flip_prob: # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                dimg1 = dimg1[:, ::-1]
                dimg2 = dimg2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if numpy.random.rand() < self.v_flip_prob: # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                dimg1 = dimg1[::-1, :]
                dimg2 = dimg2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        y0 = numpy.random.randint(0, img1.shape[0] - self.crop_size[0])
        x0 = numpy.random.randint(0, img1.shape[1] - self.crop_size[1])
        
        img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        dimg1 = dimg1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        dimg2 = dimg2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        return img1, img2,dimg1,dimg2, flow

    def __call__(self, img1, img2, dimg1, dimg2, flow):
        # img1, img2, dimg1, dimg2 = self.eraser_transform(img1, img2, dimg1, dimg2)
        img1, img2, dimg1, dimg2 ,flow= self.spatial_transform(img1, img2, dimg1, dimg2, flow)

        img1 = numpy.ascontiguousarray(img1)
        img2 = numpy.ascontiguousarray(img2)
        dimg1 = numpy.ascontiguousarray(dimg1)
        dimg2 = numpy.ascontiguousarray(dimg2)
        flow = numpy.ascontiguousarray(flow)

        return img1, img2, dimg1, dimg2, flow


class EventSequence(object):
    def __init__(self, dataframe, params, features=None, timestamp_multiplier=None, convert_to_relative=False):
        if isinstance(dataframe, pandas.DataFrame):
            self.feature_names = dataframe.columns.values
            self.features = dataframe.to_numpy()
        else:
            self.feature_names = numpy.array(['ts', 'x', 'y', 'p'], dtype=object)
            if features is None:
                self.features = numpy.zeros([1, 4])
            else:
                self.features = features
        self.image_height = params['height']
        self.image_width = params['width']
        if not self.is_sorted():
            self.sort_by_timestamp()
        if timestamp_multiplier is not None:
            self.features[:,0] *= timestamp_multiplier
        if convert_to_relative:
            self.absolute_time_to_relative()

    def get_sequence_only(self):
        return self.features

    def __len__(self):
        return len(self.features)

    def __add__(self, sequence):
        event_sequence = EventSequence(dataframe=None,
                                       features=numpy.concatenate([self.features, sequence.features]),
                                       params={'height': self.image_height,
                                               'width': self.image_width})
        return event_sequence

    def is_sorted(self):
        return numpy.all(self.features[:-1, 0] <= self.features[1:, 0])

    def sort_by_timestamp(self):
        if len(self.features[:, 0]) > 0:
            sort_indices = numpy.argsort(self.features[:, 0])
            self.features = self.features[sort_indices]

    def absolute_time_to_relative(self):
        """Transforms absolute time to time relative to the first event."""
        start_ts = self.features[:,0].min()
        assert(start_ts == self.features[0,0])
        self.features[:,0] -= start_ts

def vis_map_RGB(map, save_map_path, name):
    channel,h,w = map.shape
    if(channel==5): # events
        map = numpy.concatenate([map, numpy.zeros((1,h,w))], axis=0)
        map_img1 = map[:3, ...]
        map_img2 = map[3:, ...]
        for c in range(3):
            if(map_img1[c].mean() != 0):
                map_img1[c] = (map_img1[c] - map_img1[c].min()) / (map_img1[c].max() - map_img1[c].min()) * 255
            if(map_img2[c].mean() != 0):
                map_img2[c] = (map_img2[c] - map_img2[c].min()) / (map_img2[c].max() - map_img2[c].min()) * 255

        map_img_sum = numpy.concatenate([map_img1, map_img2], axis=2) # 叠加在
        map_img_sum = numpy.asarray(map_img_sum.transpose(1,2,0), dtype=numpy.uint8)
    elif(channel==3):
        map_img = (map - numpy.min(map)) / (numpy.max(map) - numpy.min(map)) * 255
        map_img_sum = numpy.asarray(map_img.transpose(1,2,0), dtype=numpy.uint8)

    elif(channel>5):
        map_img = numpy.squeeze(numpy.sum(map, axis=0))
        if(map_img.mean() != 0):
            map_img = (map_img - map_img.min()) / (map_img.max() - map_img.min())
        map_img_sum = numpy.asarray(map_img * 255, dtype=numpy.uint8)

    if not os.path.exists(save_map_path):
        os.makedirs(save_map_path)
    cv2.imwrite(os.path.join(save_map_path, name), map_img_sum)
    return 


class EventSequenceToVoxelGrid_Pytorch(object):
    # Source: https://github.com/uzh-rpg/rpg_e2vid/blob/master/utils/inference_utils.py#L480
    def __init__(self, num_bins, gpu=False, gpu_nr=0, normalize=True, forkserver=True, pol=True):
        if forkserver:
            try:
                torch.multiprocessing.set_start_method('forkserver')
            except RuntimeError:
                pass
        self.num_bins = num_bins
        self.normalize = normalize
        self.pol = pol
        if gpu:
            if not torch.cuda.is_available():
                print('Warning: There\'s no CUDA support on this machine!')
            else:
                self.device = torch.device('cuda:' + str(gpu_nr))
        else:
            self.device = torch.device('cpu')

    def __call__(self, event_sequence):
        """
        Build a voxel grid with bilinear interpolation in the time domain from a set of events.
        :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
        :param num_bins: number of bins in the temporal axis of the voxel grid
        :param width, height: dimensions of the voxel grid
        :param device: device to use to perform computations
        :return voxel_grid: PyTorch event tensor (on the device specified)
        """

        events = event_sequence.features.astype('float')

        width = event_sequence.image_width
        height = event_sequence.image_height

        assert (events.shape[1] == 4)
        assert (self.num_bins > 0)
        assert (width > 0)
        assert (height > 0)

        with torch.no_grad():

            events_torch = torch.from_numpy(events)
            # with DeviceTimer('Events -> Device (voxel grid)'):
            events_torch = events_torch.to(self.device)

            # with DeviceTimer('Voxel grid voting'):
            voxel_grid = torch.zeros(self.num_bins, height, width, dtype=torch.float32, device=self.device).flatten()

            # normalize the event timestamps so that they lie between 0 and num_bins
            last_stamp = events_torch[-1, 0]
            first_stamp = events_torch[0, 0]

            assert last_stamp.dtype == torch.float64, 'Timestamps must be float64!'
            # assert last_stamp.item()%1 == 0, 'Timestamps should not have decimals'

            deltaT = last_stamp - first_stamp

            if deltaT == 0:
                deltaT = 1.0

            events_torch[:, 0] = (self.num_bins - 1) * (events_torch[:, 0] - first_stamp) / deltaT
            ts = events_torch[:, 0]
            xs = events_torch[:, 1].long()
            ys = events_torch[:, 2].long()
            pols = events_torch[:, 3].float()
            if self.pol: # polarity in voxel grid
                pols[pols == 0] = -1  # polarity should be +1 / -1


                tis = torch.floor(ts)
                tis_long = tis.long()
                dts = ts - tis
                vals_left = pols * (1.0 - dts.float())
                vals_right = pols * dts.float()

                valid_indices = tis < self.num_bins
                valid_indices &= tis >= 0

                if events_torch.is_cuda:
                    datatype = torch.cuda.LongTensor
                else:
                    datatype = torch.LongTensor

                voxel_grid.index_add_(dim=0,
                                      index=(xs[valid_indices] + ys[valid_indices]
                                             * width + tis_long[valid_indices] * width * height).type(
                                          datatype),
                                      source=vals_left[valid_indices])


                valid_indices = (tis + 1) < self.num_bins
                valid_indices &= tis >= 0

                voxel_grid.index_add_(dim=0,
                                      index=(xs[valid_indices] + ys[valid_indices] * width
                                             + (tis_long[valid_indices] + 1) * width * height).type(datatype),
                                      source=vals_right[valid_indices])

                voxel_grid = voxel_grid.view(self.num_bins, height, width)
            else: #polarity in channels
                voxel_grid = torch.zeros(self.num_bins, height, width, 2, dtype=torch.float32,
                                         device=self.device).flatten(end_dim=2)
                pols[pols == 0] = -1  # polarity should be +1 / -1
                mask_pos = (pols == 1)
                mask_neg = (pols == -1)

                tis = torch.floor(ts)
                tis_long = tis.long()
                dts = ts - tis
                vals_left = 1.0 - dts.float()
                vals_right = dts.float()

                valid_indices = tis < self.num_bins
                valid_indices &= tis >= 0

                if events_torch.is_cuda:
                    datatype = torch.cuda.LongTensor
                else:
                    datatype = torch.LongTensor

                voxel_grid[:,0].index_add_(dim=0,
                                      index=(xs[valid_indices*mask_pos] + ys[valid_indices*mask_pos]
                                             * width + tis_long[valid_indices*mask_pos] * width * height).type(
                                          datatype),
                                      source=vals_left[valid_indices*mask_pos])
                voxel_grid[:,1].index_add_(dim=0,
                                           index=(xs[valid_indices * mask_neg] + ys[valid_indices * mask_neg]
                                                  * width + tis_long[valid_indices * mask_neg] * width * height).type(
                                               datatype),
                                           source=vals_left[valid_indices*mask_neg])

                valid_indices = (tis + 1) < self.num_bins
                valid_indices &= tis >= 0

                voxel_grid[:,0].index_add_(dim=0,
                                      index=(xs[valid_indices*mask_pos] + ys[valid_indices*mask_pos] * width
                                             + (tis_long[valid_indices*mask_pos] + 1) * width * height).type(datatype),
                                      source=vals_right[valid_indices*mask_pos])
                voxel_grid[:,1].index_add_(dim=0,
                                      index=(xs[valid_indices*mask_neg] + ys[valid_indices*mask_neg] * width
                                             + (tis_long[valid_indices*mask_neg] + 1) * width * height).type(datatype),
                                      source=vals_right[valid_indices*mask_neg])

                voxel_grid = voxel_grid.view(self.num_bins, height, width,2).permute(0,3,1,2)


        if self.normalize:
            mask = torch.nonzero(voxel_grid, as_tuple=True)
            if mask[0].size()[0] > 0:
                mean = voxel_grid[mask].mean()
                std = voxel_grid[mask].std()
                if std > 0:
                    voxel_grid[mask] = (voxel_grid[mask] - mean) / std
                else:
                    voxel_grid[mask] = voxel_grid[mask] - mean

        return voxel_grid

