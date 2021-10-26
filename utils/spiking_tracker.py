import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.localization_utils import center_error, format_bbox, iou
from utils.misc import tonp
from utils.neural_coding import saccade_coding
import numpy as np
import got10k

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class SpikingTracker(object):
    def __init__(self, model, criterion, optimizer, is_training=True, is_recording=True):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.is_training = is_training
        self.is_recording = is_recording

    def first_frame(self, frame: torch.Tensor, bbox, seq_name: str, args):
        # shape [1, T, C, H, W]
        saccades = saccade_coding(
            frame, timesteps=args.timesteps, delta_threshold=0.2)

        loss_tv = torch.tensor(0.).to(device)

        for k in range(saccades.shape[0]):
            s, r, u = self.model(saccades[0])
            loss_ = self.criterion(s, r, u, target=bbox,  sum_=False)

            loss_tv += sum(loss_)

            # ONLINE LEARNING UPDATE
            if self.is_training:
                loss_tv.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            # reinitialize loss_tv
            loss_tv = torch.tensor(0.).to(device)

        # get the original IoU
        r_np = np.array(tonp(r))
        IoU, CE = self._compute_metrics(r_np[-1], bbox, args)

        print(
            f'Initialized the tracker of sequence name="{seq_name}" with IoU={IoU} and CE={CE}')

        # initialize the profiling states
        self.counts = [0.0 for _ in range(len(self.model))]
        self.activities = [0.0 for _ in range(len(self.model))]
        self.current_ts = 0
        self.CEs = []
        self.IoUs = []

    def update(self, event_frame: torch.Tensor, gt_bbox: torch.Tensor, seq_name: str, args):
        loss_tv = torch.tensor(0.).to(device)

        s, r, u = self.model(event_frame)

        loss_ = self.criterion(s, r, u, target=gt_bbox, sum_=False)

        loss_tv += sum(loss_)

        # ONLINE LEARNING UPDATE
        if self.is_training:
            loss_tv.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        r_np = np.array(tonp(r))

        # compute iou
        IoU, CE = self._compute_metrics(r_np[-1], gt_bbox, args)

        # update profiling states
        self._spike_count(s.detach().cpu(), u.detach().cpu())
        self.IoUs.append(IoU)
        self.CEs.append(CE)
        self.current_ts += 1

        # returns the bbox of the last layer (it's the final prediction)
        return r_np[-1]

    def _spike_count(self, s, u):
        """Counts the spikes in the SNN to measure the activity rate

        Args:
            s (torch.Tensor): the output spikes per layer
            u (torch.Tensor): the membrane potentials per layer
        """

        for i in range(len(self.counts) - 1):
            self.counts[i] += s[i]
            self.activities[i] += tonp(s[i].mean().data)

        # because s[-1] has no threshold dynamic in the original implementation
        self.counts[-1] += (u[-1] >= 0.).float()
        self.activities[-1] += tonp((u[-1] >= 0.).mean().data)

    def get_activities(self):
        result = []
        for activity in self.activities:
            result.append(activity / self.current_ts)

        return result

    def _compute_metrics(self, pred_bbox, gt_bbox, args):
        # clip between 0 and 1
        pred_bbox = format_bbox(pred_bbox, args.height, args.width)
        gt_bbox = format_bbox(gt_bbox, args.height, args.width)

        # compute iou
        IoU = iou(pred_bbox, gt_bbox)
        CE = center_error(pred_bbox, gt_bbox)

        return IoU, CE

    def profile(self, epoch, seq_name, args):
        prefix = 'TRAIN' if self.is_training else 'VAL'

        if self.is_recording:
            pass

    def _calc_curves(self):
        ious = np.asarray(self.IoUs, float)[:, np.newaxis]
        center_errors = np.asarray(self.CEs, float)[:, np.newaxis]

        thr_iou = np.linspace(0, 1, self.nbins_iou)[np.newaxis, :]
        thr_ce = np.arange(0, self.nbins_ce)[np.newaxis, :]

        bin_iou = np.greater(ious, thr_iou)
        bin_ce = np.less_equal(center_errors, thr_ce)

        succ_curve = np.mean(bin_iou, axis=0)
        prec_curve = np.mean(bin_ce, axis=0)

        return succ_curve, prec_curve