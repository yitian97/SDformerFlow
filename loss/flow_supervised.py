import torch
import numpy as np
import math
MAX_FLOW = 400

class flow_loss_supervised(torch.nn.Module):
    def __init__(self,  config, device):
        super().__init__()
        self.device = device
        self.flow_scaling = config['metrics']['flow_scaling']
        self.lambda_mod = config["loss"]["lambda_mod"]
        self.lambda_ang = config["loss"]["lambda_ang"]

    def mod_loss_function(self,flow,gt_flow,mask,num_valid_px):

        # flow *= self._dt_gt.to(self.device) / self._dt_input.to(self.device)

        # compute AEE
        error = torch.sqrt((flow - gt_flow).pow(2).sum(1)+ 1e-8)

        # mask AEE and flow
        # mask = torch.squeeze(mask, 1)

        mask = mask.reshape(flow.shape[0], -1)

        error = error.view(flow.shape[0], -1)
        error = error * mask



        return  torch.sum(error, dim=1) / (num_valid_px + 1e-9)
    def angular_loss_function(self,flow,gt_flow,mask,num_valid_px,epsilon=1e-8):
        flow_mag = torch.sqrt(flow.pow(2).sum(1)+epsilon)
        gt_mag =  torch.sqrt(gt_flow.pow(2).sum(1)+epsilon)
        dot_product = flow[:, 0] * gt_flow[:, 0] + flow[:, 1] * gt_flow[:, 1]
        cosine = (dot_product + epsilon) / (flow_mag * gt_mag + epsilon)
        # cosine = (dot_product) / (pred_mod*label_mod)
        cosine = torch.clamp(cosine, min=-1. + epsilon, max=1. - epsilon)
        return torch.sum(torch.acos(cosine) * mask) /  (num_valid_px + 1e-9)

    def rel_loss_function(self,flow,gt_flow,mask,num_valid_px, epsilon=1e-7):
        error = (flow - gt_flow).pow(2).sum(1).sqrt()
        gt_mag = gt_flow.pow(2).sum(1).sqrt()

        return (1 / (num_valid_px + 1e-9)) * torch.sum((error * mask) / (gt_mag + epsilon))

    def cosine_loss_function(self,flow,gt_flow,mask,num_valid_px, epsilon=1e-7):
        flow_mag = flow.pow(2).sum(1).sqrt()
        gt_mag = gt_flow.pow(2).sum(1).sqrt()
        dot_product = flow[:, 0] * gt_flow[:, 0] + flow[:, 1] * gt_flow[:, 1]
        cosine = (dot_product + epsilon) / (flow_mag * gt_mag + epsilon)
        cosine = torch.clamp(cosine, min=-1. + epsilon, max=1. - epsilon)

        return torch.sum((1. - cosine) * mask) / (num_valid_px + 1e-9)

    def sequence_loss(self, flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
        """ Loss function defined over sequence of flow predictions """
        n_predictions = len(flow_preds)
        flow_loss = 0.0

        # exlude invalid pixels and extremely large diplacements
        mag = torch.sum(flow_gt ** 2, dim=1).sqrt()  # b,h,w
        valid = (valid >= 0.5) & (mag < max_flow)  # b,1,h,w

        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i - 1)
            i_loss = (flow_preds[i] - flow_gt).abs()
            flow_loss += i_weight * (valid[:, None] * i_loss).mean()

        # epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
        # epe = epe.view(-1)[valid.view(-1)]

        # metrics = {
        #     'epe': epe.mean().item(),
        #     '1px': (epe < 1).float().mean().item(),
        #     '3px': (epe < 3).float().mean().item(),
        #     '5px': (epe < 5).float().mean().item(),
        # }

        return flow_loss
    def forward(self, pred_list, gt_flow, mask, gamma=None):

        # flow B 2 H W
        # flow mag B H W
        # error B H W
        # convert flow
        mod_loss = 0.
        ang_loss = 0.
        curr_loss = 0.
        if gamma is not None:
            curr_loss = self.sequence_loss(pred_list, gt_flow, mask, gamma)
        else:
            num_valid_px = torch.sum(mask)

            for pred in pred_list:
                flow = pred * self.flow_scaling
                mod_loss = self.mod_loss_function(flow,gt_flow,mask,num_valid_px)
                # ang_loss = self.angular_loss_function(flow, gt_flow, mask, num_valid_px)
                # curr_loss += self.lambda_mod * mod_loss + self.lambda_ang * ang_loss
                curr_loss += self.lambda_mod * mod_loss
            curr_loss = curr_loss/len(pred_list)
            curr_loss = torch.mean(curr_loss)


        return  curr_loss


class AEE(torch.nn.Module):

    def __init__(self, pred, label, mask, flow_scaling=128):
        super().__init__()
        self.flow = pred
        self.label = label
        self.mask = mask
        self.flow_scaling = flow_scaling



    def forward(self):

        # flow B 2 H W
        # flow mag B H W
        # error B H W
        # convert flow
        flow = self.flow * self.flow_scaling
        # flow *= self._dt_gt.to(self.device) / self._dt_input.to(self.device)
        flow_mag = flow.pow(2).sum(1).sqrt()

        # compute AEE
        error = (flow - self.label).pow(2).sum(1).sqrt()
        mask = self.mask.reshape(flow.shape[0], -1)
        error = error.view(flow.shape[0], -1)
        flow_mag = flow_mag.view(flow.shape[0], -1)
        error = error * mask
        flow_mag = flow_mag * mask
        # compute AEE and percentage of outliers
        num_valid_px = torch.sum(mask, dim=1)
        AEE = torch.sum(error, dim=1) / (num_valid_px + 1e-9)

        outliers = (error > 3.0) * (error > 0.05 * flow_mag)  # AEE larger than 3px and 5% of the flow magnitude
        percent_AEE = outliers.sum() / (num_valid_px + 1e-9)
        outliers_p1 =  error > 1.0
        outliers_p2 =  error > 2.0
        outliers_p3 =  error > 3.0
        PE1 = outliers_p1.sum() / (num_valid_px + 1e-9)
        PE2 = outliers_p2.sum() / (num_valid_px + 1e-9)
        PE3 = outliers_p3.sum() / (num_valid_px + 1e-9)

        return AEE, PE1, PE2, PE3, percent_AEE


class AAE(torch.nn.Module):

    def __init__(self, pred, label, mask, flow_scaling=128):
        super().__init__()
        self.flow = pred
        self.label = label
        self.mask = mask
        self.flow_scaling = flow_scaling



    def forward(self):
        flow = self.flow * self.flow_scaling
        flow_mag = flow.pow(2).sum(1).sqrt()
        flow_mag = flow_mag * self.mask
        gt_mag = self.label.pow(2).sum(1).sqrt()
        gt_mag = gt_mag * self.mask
        num_valid_px = torch.sum(self.mask)
        dot_product = flow[:, 0] * self.label[:, 0] + flow[:, 1] * self.label[:, 1]
        cosine = (dot_product + 1e-7) / (flow_mag * gt_mag + 1e-7)
        cosine = torch.clamp(cosine, min=-1. + 1e-7, max=1. - 1e-7)
        AAE =  torch.sum(torch.acos(cosine) * self.mask) / num_valid_px

        return  AAE* 180 / math.pi,