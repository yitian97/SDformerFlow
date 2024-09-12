import torch
import torch.nn.functional as F
from math import sin, cos, pi
import numbers
import random
import torchvision.transforms as TvT
"""
    Data augmentation functions.
    There are some problems with torchvision data augmentation functions:
    1. they work only on PIL images, which means they cannot be applied to tensors with more than 3 channels,
       and they require a lot of conversion from Numpy -> PIL -> Tensor
    2. they do not provide access to the internal transformations (affine matrices) used, which prevent
       applying them for more complex tasks, such as transformation of an optic flow field (for which
       the inverse transformation must be known).
    For these reasons, we implement my own data augmentation functions
    (strongly inspired by torchvision transforms) that operate directly
    on Torch Tensor variables, and that allow to transform an optic flow field as well.
"""


class Compose(object):
    """
    Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        # >>> transforms.Compose([
        # >>>     transforms.CenterCrop(10),
        # >>>     transforms.ToTensor(),
        # >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class CenterCrop(object):
    """Center crop the tensor to a certain size.
    """

    def __init__(self, size, preserve_mosaicing_pattern=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        self.preserve_mosaicing_pattern = preserve_mosaicing_pattern

    def __call__(self, x):
        """
            x: [C x H x W] Tensor to be rotated.
            is_flow: this parameter does not have any effect
        Returns:
            Tensor: Cropped tensor.
        """
        events,flow,mask = x
        h, w = events.shape[-2], events.shape[-1]
        th, tw = self.size
        assert(th <= h)
        assert(tw <= w)
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))

        if self.preserve_mosaicing_pattern:
            # make sure that i and j are even, to preserve
            # the mosaicing pattern
            if i % 2 == 1:
                i = i + 1
            if j % 2 == 1:
                j = j + 1


        return (events[..., i:i + th, j:j + tw],flow[:,:, i:i + th, j:j + tw],mask[: ,:, i:i + th, j:j + tw])

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomCrop(object):
    """Crop the tensor at a random location.
    """

    def __init__(self, size, preserve_mosaicing_pattern=False):

        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        self.preserve_mosaicing_pattern = preserve_mosaicing_pattern

    @staticmethod
    def get_params(x, output_size):
        h, w = x.shape[-2], x.shape[-1]
        th, tw = output_size
        assert(th <= h)
        assert(tw <= w)
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        return i, j, th, tw

    def __call__(self, x):
        """
            x: [B X C x H x W] Tensor to be rotated.
            is_flow: this parameter does not have any effect
        Returns:
            Tensor: Cropped tensor.

        """
        events, flow, mask = x
        i, j, h, w = self.get_params(events, self.size)

        if self.preserve_mosaicing_pattern:
            # make sure that i and j are even, to preserve the mosaicing pattern
            if i % 2 == 1:
                i = i + 1
            if j % 2 == 1:
                j = j + 1

        return (events[..., i:i + h, j:j + w],flow[:, :, i:i + h, j:j + w],mask[:, :, i:i + h, j:j + w])

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomRotationFlip(object):
    """Rotate the image by angle.
    """

    def __init__(self, degrees, p_hflip=0.5, p_vflip=0.5):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.p_hflip = p_hflip
        self.p_vflip = p_vflip

    @staticmethod
    def get_params(degrees, p_hflip, p_vflip):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])
        angle_rad = angle * pi / 180.0

        M_original_transformed = torch.FloatTensor([[cos(angle_rad), -sin(angle_rad), 0],
                                                    [sin(angle_rad), cos(angle_rad), 0],
                                                    [0, 0, 1]])

        if random.random() < p_hflip:
            M_original_transformed[:, 0] *= -1

        if random.random() < p_vflip:
            M_original_transformed[:, 1] *= -1

        M_transformed_original = torch.inverse(M_original_transformed)

        M_original_transformed = M_original_transformed[:2, :].unsqueeze(dim=0)  # 3 x 3 -> N x 2 x 3
        M_transformed_original = M_transformed_original[:2, :].unsqueeze(dim=0)

        return M_original_transformed, M_transformed_original

    def __call__(self, x):
        """
            x: [B, C x H x W] Tensor to be rotated.
            is_flow: if True, x is an [2 x H x W] displacement field, which will also be transformed
        Returns:
            Tensor: Rotated tensor.
        """

        output1,output2,output3 = [],[],[]
        events, flow, mask = x
        for batch_id in range(events.shape[0]):


            M_original_transformed, M_transformed_original = self.get_params(self.degrees, self.p_hflip, self.p_vflip)
            affine_grid = F.affine_grid(M_original_transformed, events[batch_id].unsqueeze(dim=0).shape, align_corners=False)
            transformed_events = F.grid_sample(events[batch_id].unsqueeze(dim=0), affine_grid, align_corners=False)
            transformed_flow = F.grid_sample(flow[batch_id].unsqueeze(dim=0), affine_grid, align_corners=False)
            transformed_mask = F.grid_sample(mask[batch_id].unsqueeze(dim=0).float(), affine_grid, align_corners=False)
            # if is_flow:
                # Apply the same transformation to the flow field
            A00 = M_transformed_original[0, 0, 0]
            A01 = M_transformed_original[0, 0, 1]
            A10 = M_transformed_original[0, 1, 0]
            A11 = M_transformed_original[0, 1, 1]
            vx = transformed_flow[:, 0, :, :].clone()
            vy = transformed_flow[:, 1, :, :].clone()
            transformed_flow[:, 0, :, :] = A00 * vx + A01 * vy
            transformed_flow[:, 1, :, :] = A10 * vx + A11 * vy

            output1.append(transformed_events)
            output2.append(transformed_flow)
            output3.append(transformed_mask.bool())

        return (torch.cat(output1, dim=0),torch.cat(output2, dim=0),torch.cat(output3, dim=0))

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', p_flip={:.2f}'.format(self.p_hflip)
        format_string += ', p_vlip={:.2f}'.format(self.p_vflip)
        format_string += ')'
        return format_string

class Random_horizontal_flip(object):

    def __init__(self, p = 0.5):

        self.p = p

    def __call__(self, x):

        events, flow, mask = x

        if torch.rand(1).item() <= self.p:

            events = TvT.functional.hflip(events)

            flow = TvT.functional.hflip(flow)
            flow[:,0] *= -1

            mask = TvT.functional.hflip(mask.float())


        return (events, flow, mask.bool())

class Random_vertical_flip(object):

    def __init__(self, p = 0.5):

        self.p = p

    def __call__(self, x):

        events, flow, mask = x

        if torch.rand(1).item() <= self.p:

            events = TvT.functional.vflip(events)

            flow = TvT.functional.vflip(flow)
            flow[:,1] *= -1

            mask = TvT.functional.vflip(mask.float())

        return (events, flow, mask.bool())

class Random_event_drop(object):

    def __init__(self, min_drop_rate=0., max_drop_rate=0.6, p=0.5):
        self.p = p
        self.min_drop_rate = min_drop_rate
        self.max_drop_rate = max_drop_rate

    def __call__(self, x):
        events, flow, mask = x

        if torch.rand(1).item() <= self.p:
            # probability of an input event to be dropped: random variable uniformly distributed on [TBD, TBD] by default
            q = (self.min_drop_rate - self.max_drop_rate) * torch.rand(1) + self.max_drop_rate

            ev_mask = torch.rand_like(events)
            events = events * (ev_mask > q)

        return (events, flow, mask)


def downsample_data(x,scale_factor):
    output = ()
    for input in x:
        # input= input.unsqueeze(0)
        input= F.interpolate(input, scale_factor=scale_factor, mode='bilinear',
                             recompute_scale_factor=False, align_corners=False)
        output += (input,)
    return output



if __name__ == "__main__":

    chunk = torch.randn( 1, 10, 480, 640)
    label = torch.randn( 2, 2, 480, 640)
    mask = torch.randn_like(label)


    data_augmentation = Compose([
        RandomCrop((288,384)),
        # Random_vertical_flip(0.5),
        # Random_horizontal_flip(0.5),
        # Random_event_drop(0,0.6),
        # CenterCrop(200,200)
        # RandomRotationFlip((0,0),0.5,0.5),
    ])

    # chunk, label, mask = downsample_data((chunk, label, mask),0.5)
    print(torch.max(chunk))
    print(torch.min(chunk))
    chunk,label,mask = data_augmentation((chunk, label, mask))
    print(torch.max(chunk))
    print(torch.min(chunk))
    print()


