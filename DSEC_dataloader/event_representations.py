import numpy as np
import math
from numba import jit

import os
import torch
#
import hdf5plugin
os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH

import tqdm
import h5py
from typing import Dict, Tuple
'''
Adpated from DSEC GIT REPOSITORY
https://github.com/uzh-rpg/DSEC
'''


def rectify_events(x: np.ndarray, y: np.ndarray, rectify_map):


    height = 480; width = 640

    assert rectify_map.shape == (height, width, 2), rectify_map.shape
    assert x.max() < width
    assert y.max() < height
    return rectify_map[y, x]

def bilinear_sample(x,y):
    k = max(0,1-(x-y).abs())
    return k
def cumulate_spikes_into_frames(X_list, Y_list, P_list):

    frame = np.zeros((2, 480, 640), dtype='float')

    for x, y, p in zip(X_list, Y_list, P_list):
        if p == 1:
            frame[0, y, x] += 1  # register ON event on channel 0
        else:
            frame[1, y, x] += 1  # register OFF event on channel 1

    return frame

def events_to_frames(p, x, y, t, num_frames_per_ts):
        event_repr = torch.zeros((num_frames_per_ts, 2, 480, 640), dtype=torch.float)
        t= torch.from_numpy(t.astype('float64'))
        p = torch.from_numpy(p.astype('int32'))
        x = torch.from_numpy(x.astype('float32'))
        y = torch.from_numpy(y.astype('float32'))

        t_norm = (t - t[0]) / (t[-1] - t[0])
        ts_norm = ((num_frames_per_ts-1) * t_norm).int()
        assert (torch.min(ts_norm) >= 0) & (torch.max(ts_norm) < num_frames_per_ts), ''

        x0 = x.int()
        y0 = y.int()

        for xlim in [x0, x0 + 1]:
            for ylim in [y0, y0 + 1]:
                valid_mask = (xlim < 640) & (xlim >= 0) & (ylim < 480) & (ylim >= 0)
                interp_weights = torch.nn.functional.relu(1 - (xlim.float() - x).abs()) \
                                 * torch.nn.functional.relu(1 - (ylim.float() - y).abs())

                index = 640 * 480 * ts_norm.long() + 640 * ylim.long() + xlim.long()

                # event_repr[:, 0].put_(index[mask], interp_weights[mask], accumulate=True)
                pol_mask = (p == 1)
                event_repr[:, 0].put_(index[pol_mask & valid_mask], interp_weights[pol_mask & valid_mask],
                                      accumulate=True)
                pol_mask = (p == 0)
                event_repr[:, 1].put_(index[pol_mask & valid_mask], interp_weights[pol_mask & valid_mask],
                                      accumulate=True)


        return event_repr


class EventSlicer:
    def __init__(self, h5f: h5py.File):

        self.h5f = h5f

        self.events = dict()

        for dset_str in ['p', 'x', 'y', 't']:
            self.events[dset_str] = self.h5f['events/{}'.format(dset_str)]

        # This is the mapping from milliseconds to event index:
        # It is defined such that
        # (1) t[ms_to_idx[ms]] >= ms*1000
        # (2) t[ms_to_idx[ms] - 1] < ms*1000
        # ,where 'ms' is the time in milliseconds and 't' the event timestamps in microseconds.
        #
        # As an example, given 't' and 'ms':
        # t:    0     500    2100    5000    5000    7100    7200    7200    8100    9000
        # ms:   0       1       2       3       4       5       6       7       8       9
        #
        # we get
        #
        # ms_to_idx:
        #       0       2       2       3       3       3       5       5       8       9

        self.ms_to_idx = np.asarray(self.h5f['ms_to_idx'], dtype='int64')

        if "t_offset" in list(h5f.keys()):
            self.t_offset = int(h5f['t_offset'][()])
        else:
            self.t_offset = 0
        self.t_final = int(self.events['t'][-1]) + self.t_offset

    def get_start_time_us(self):
        return self.t_offset

    def get_final_time_us(self):
        return self.t_final

    def get_events(self, t_start_us: int, t_end_us: int) -> Dict[str, np.ndarray]:
        """Get events (p, x, y, t) within the specified time window
        Parameters
        ----------
        t_start_us: start time in microseconds
        t_end_us: end time in microseconds
        Returns
        -------
        events: dictionary of (p, x, y, t) or None if the time window cannot be retrieved
        """
        assert t_start_us < t_end_us

        # We assume that the times are top-off-day, hence subtract offset:
        t_start_us -= self.t_offset
        t_end_us -= self.t_offset

        t_start_ms, t_end_ms = self.get_conservative_window_ms(t_start_us, t_end_us)
        t_start_ms_idx = self.ms2idx(t_start_ms)
        t_end_ms_idx = self.ms2idx(t_end_ms)

        if t_start_ms_idx is None or t_end_ms_idx is None:
            # Cannot guarantee window size anymore
            return None

        events = dict()
        time_array_conservative = np.asarray(self.events['t'][t_start_ms_idx:t_end_ms_idx])
        idx_start_offset, idx_end_offset = self.get_time_indices_offsets(time_array_conservative, t_start_us, t_end_us)
        t_start_us_idx = t_start_ms_idx + idx_start_offset
        t_end_us_idx = t_start_ms_idx + idx_end_offset
        # Again add t_offset to get gps time
        events['t'] = time_array_conservative[idx_start_offset:idx_end_offset] + self.t_offset
        for dset_str in ['p', 'x', 'y']:
            events[dset_str] = np.asarray(self.events[dset_str][t_start_us_idx:t_end_us_idx])
            assert events[dset_str].size == events['t'].size
        return events


    @staticmethod
    def get_conservative_window_ms(ts_start_us: int, ts_end_us) -> Tuple[int, int]:
        """Compute a conservative time window of time with millisecond resolution.
        We have a time to index mapping for each millisecond. Hence, we need
        to compute the lower and upper millisecond to retrieve events.
        Parameters
        ----------
        ts_start_us:    start time in microseconds
        ts_end_us:      end time in microseconds
        Returns
        -------
        window_start_ms:    conservative start time in milliseconds
        window_end_ms:      conservative end time in milliseconds
        """
        assert ts_end_us > ts_start_us
        window_start_ms = math.floor(ts_start_us/1000)
        window_end_ms = math.ceil(ts_end_us/1000)
        return window_start_ms, window_end_ms

    @staticmethod
    @jit(nopython=True)
    def get_time_indices_offsets(
            time_array: np.ndarray,
            time_start_us: int,
            time_end_us: int) -> Tuple[int, int]:
        """Compute index offset of start and end timestamps in microseconds
        Parameters
        ----------
        time_array:     timestamps (in us) of the events
        time_start_us:  start timestamp (in us)
        time_end_us:    end timestamp (in us)
        Returns
        -------
        idx_start:  Index within this array corresponding to time_start_us
        idx_end:    Index within this array corresponding to time_end_us
        such that (in non-edge cases)
        time_array[idx_start] >= time_start_us
        time_array[idx_end] >= time_end_us
        time_array[idx_start - 1] < time_start_us
        time_array[idx_end - 1] < time_end_us
        this means that
        time_start_us <= time_array[idx_start:idx_end] < time_end_us
        """

        assert time_array.ndim == 1

        idx_start = -1
        if time_array[-1] < time_start_us:
            # This can happen in extreme corner cases. E.g.
            # time_array[0] = 1016
            # time_array[-1] = 1984
            # time_start_us = 1990
            # time_end_us = 2000

            # Return same index twice: array[x:x] is empty.
            return time_array.size, time_array.size
        else:
            for idx_from_start in range(0, time_array.size, 1):
                if time_array[idx_from_start] >= time_start_us:
                    idx_start = idx_from_start
                    break
        assert idx_start >= 0

        idx_end = time_array.size
        for idx_from_end in range(time_array.size - 1, -1, -1):
            if time_array[idx_from_end] >= time_end_us:
                idx_end = idx_from_end
            else:
                break

        assert time_array[idx_start] >= time_start_us
        if idx_end < time_array.size:
            assert time_array[idx_end] >= time_end_us
        if idx_start > 0:
            assert time_array[idx_start - 1] < time_start_us
        if idx_end > 0:
            assert time_array[idx_end - 1] < time_end_us
        return idx_start, idx_end

    def ms2idx(self, time_ms: int) -> int:
        assert time_ms >= 0
        if time_ms >= self.ms_to_idx.size:
            return None
        return self.ms_to_idx[time_ms]


class VoxelGrid:
    def __init__(self, input_size: tuple):
        assert len(input_size) == 3
        self.voxel_grid = torch.zeros((input_size), dtype=torch.float, requires_grad=False)
        self.nb_channels = input_size[0]


    def convert_CHW(self, events):
        C, H, W = self.voxel_grid.shape
        with torch.no_grad():
            self.voxel_grid = self.voxel_grid.to(events['p'].device)
            voxel_grid = self.voxel_grid.clone()

            t_norm = events['t']
            t_norm = (C - 1) * (t_norm-t_norm[0]) / (t_norm[-1]-t_norm[0]) #0-bin

            x0 = events['x'].int()
            y0 = events['y'].int()
            t0 = t_norm.int() #round

            # value = torch.abs(2*events['p']-1)
            value = 2 * events['p'] - 1

            for xlim in [x0,x0+1]:
                for ylim in [y0,y0+1]:
                    for tlim in [t0,t0+1]:

                        mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (ylim >= 0) & (tlim >= 0) & (tlim < self.nb_channels)
                        interp_weights = value * (1 - (xlim-events['x']).abs()) * (1 - (ylim-events['y']).abs()) * (1 - (tlim - t_norm).abs())

                        index = H * W * tlim.long() + \
                                W * ylim.long() + \
                                xlim.long()

                        voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

        return voxel_grid

    def convert_CHW_polarities(self, events):
        C, H, W = self.voxel_grid.shape
        with torch.no_grad():
            self.voxel_grid = self.voxel_grid.to(events['p'].device)
            voxel_grid_pos = self.voxel_grid.clone()
            voxel_grid_neg = self.voxel_grid.clone()

            t_norm = events['t']
            t_norm = (C - 1) * (t_norm-t_norm[0]) / (t_norm[-1]-t_norm[0]) #0-bin

            x0 = events['x'].int()
            y0 = events['y'].int()
            t0 = t_norm.int() #round

            # value = torch.abs(2*events['p']-1)
            # value = 2 * events['p'] - 1
            mask_pos = (events['p'] == 1).bool()
            mask_neg = (events['p'] == 0).bool()
            for xlim in [x0,x0+1]:
                for ylim in [y0,y0+1]:
                    for tlim in [t0,t0+1]:

                        mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (ylim >= 0) & (tlim >= 0) & (tlim < self.nb_channels)
                        interp_weights =   (1 - (xlim-events['x']).abs()) * (1 - (ylim-events['y']).abs()) * (1 - (tlim - t_norm).abs())

                        index = H * W * tlim.long() + \
                                W * ylim.long() + \
                                xlim.long()
                        # mask_value = events['p']

                        voxel_grid_pos.put_(index[mask*mask_pos], interp_weights[mask*mask_pos], accumulate=True)
                        voxel_grid_neg.put_(index[mask*mask_neg], interp_weights[mask*mask_neg], accumulate=True)
            voxel_grid = torch.cat((voxel_grid_pos.unsqueeze(dim=1),voxel_grid_neg.unsqueeze(dim=1)),dim=1)

        return voxel_grid

def events_to_voxel_grid_v2(events, num_bins, height, width, normalize=True):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [x, y, timestamp, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    """

    # assert (events.shape[1] == 4)
    assert (num_bins > 0)
    assert (width > 0)
    assert (height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events['t'][-1]
    first_stamp = events['t'][0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    events['t'] = (num_bins - 1) * (events['t'] - first_stamp) / deltaT
    ts = events['t']
    xs = events['x'].astype(int)
    ys = events['y'].astype(int)
    pols = events['p']
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(int)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    if normalize:
        mask = np.nonzero(voxel_grid)
        if mask[0].size > 0:
            mean, stddev = voxel_grid[mask].mean(), voxel_grid[mask].std()
            if stddev > 0:
                voxel_grid[mask] = (voxel_grid[mask] - mean) / stddev

    return voxel_grid


if __name__=='__main__':
    sequence = 'zurich_city_09_a'
    root='../data/Datasets/DSEC'
    events_input='voxel'
    num_frames_per_ts=15

    timestamps = np.loadtxt(os.path.join(root, 'train_optical_flow', sequence, 'flow', 'forward_timestamps.txt'), delimiter = ',', dtype='int64')
    # timestamps = np.loadtxt(os.path.join(root, 'test_forward_optical_flow_timestamps', sequence + ".csv"), delimiter = ',', dtype='int64')

    eventsL_path = os.path.join(root, 'train_events', sequence, 'events', 'left')
    # eventsL_path = os.path.join(root, 'test_events', sequence, 'events', 'left')


    # save_path_events = os.path.join(root, 'saved_flow_data', 'event_tensors',  '{}bins'.format(str(num_frames_per_ts).zfill(2)), 'left')

    datafile_path = os.path.join(eventsL_path, "events.h5")
    datafile = h5py.File(datafile_path, 'r')
    event_slicer = EventSlicer(datafile)

    N_chunks = timestamps.shape[0]  # N_chunks = N_grountruths

    fileidx = 0

    for numchunk in range(N_chunks):

        fileidx += 1

        t_beg, t_end = timestamps[numchunk]

        dt = (t_end - t_beg) / num_frames_per_ts
        # chunk = []
        event_data = event_slicer.get_events(t_beg, t_end)
        p = event_data['p']
        t = event_data['t']
        x = event_data['x']
        y = event_data['y']
        # rectify events
        rectmap_path = os.path.join(eventsL_path, "rectify_map.h5")
        rectmap_file = h5py.File(rectmap_path)
        rectmap = rectmap_file['rectify_map'][()]

        xy_rect = rectify_events(x, y, rectmap)
        x_rect = xy_rect[:, 0]
        y_rect = xy_rect[:, 1]




        # generate voxel
        if (events_input == "voxel"):
            voxel_grid = VoxelGrid((num_frames_per_ts, 480, 640))
            pol = p.astype('float32')
            t = t.astype('float32')

            event_data_torch = {
                'p': torch.from_numpy(pol),
                't': torch.from_numpy(t),
                'x': torch.from_numpy(x_rect),
                'y': torch.from_numpy(y_rect),
            }
            mask = (x_rect >= 0) & (x_rect < 640) & (y_rect >= 0) & (y_rect < 480)

            event_data_np = {
                'p': pol[mask],
                't': t[mask],
                'x': x_rect[mask],
                'y': y_rect[mask],
            }
            chunk1 =voxel_grid.convert_CHW(event_data_torch)
            chunk2 = events_to_voxel_grid_v2(event_data_np,num_frames_per_ts,480,640, False)
            frame = events_to_frames(pol, x_rect,  y_rect, t, num_frames_per_ts)
            print(torch.max(chunk1))
            print(torch.min(chunk1))
            print(np.max(chunk2))
            print(np.min(chunk2))
            print(torch.max(frame))
            print(torch.min(frame))
            print(torch.mean(frame[frame != 0].mean()))
            pass