from torchvision import transforms
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.utils import _log_api_usage_once

def parse_matrix(matrix_str):
    rows = matrix_str.strip().split('] [')
    matrix = []
    for row in rows:
        row = row.replace('[', '').replace(']', '')
        matrix.append(list(map(float, row.split())))
    return np.array(matrix)


def _is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError("clip should be Tensor. Got %s" % type(clip))

    if not clip.ndimension() == 4:
        raise ValueError("clip should be 4D. Got %dD" % clip.dim())

    return True

def crop(clip, i, j, h, w):
    """
    Args:
        clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
    """
    if len(clip.size()) != 4:
        raise ValueError("clip should be a 4D tensor")
    return clip[..., i : i + h, j : j + w]

def to_tensor(clip):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
    """
    _is_tensor_video_clip(clip)
    if not clip.dtype == torch.uint8:
        raise TypeError("clip tensor should have data type uint8. Got %s" % str(clip.dtype))
    # return clip.float().permute(3, 0, 1, 2) / 255.0
    return clip.float() / 255.0

def resize_with_scale_factor(clip, scale_factor, interpolation_mode):
    return torch.nn.functional.interpolate(clip, scale_factor=scale_factor, mode=interpolation_mode, align_corners=False)

def center_crop(clip, crop_size):
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    h, w = clip.size(-2), clip.size(-1)
    # print(clip.shape)
    th, tw = crop_size
    if h < th or w < tw:
        # print(h, w)
        raise ValueError("height {} and width {} must be no smaller than crop_size".format(h, w))

    i = int(round((h - th) / 2.0))
    j = int(round((w - tw) / 2.0))
    return crop(clip, i, j, th, tw), i, j

class ToTensorVideo:
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    """

    def __init__(self):
        pass

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
        """
        return to_tensor(clip)

    def __repr__(self) -> str:
        return self.__class__.__name__

class SDXLCenterCrop:
    def __init__(
        self,
        size,
        interpolation_mode="bilinear",
    ):
        if isinstance(size, tuple):
            if len(size) != 2:
                raise ValueError(f"size should be tuple (height, width), instead got {size}")
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation_mode = interpolation_mode
       

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: scale resized / center cropped video clip.
                size is (T, C, crop_size, crop_size)
        """
        # add aditional one pixel for avoiding error in center crop 
        ori_h, ori_w = clip.size(-2), clip.size(-1)
        tar_h, tar_w = self.size[0] + 1, self.size[1] + 1
        tar_h_div_ori_h = tar_h / ori_h
        tar_w_div_ori_w = tar_w / ori_w
        # print('before resize', clip.shape)
        if tar_h_div_ori_h > tar_w_div_ori_w:
            clip = resize_with_scale_factor(clip=clip, scale_factor=tar_h_div_ori_h, interpolation_mode=self.interpolation_mode)
            # print('after h resize', clip.shape)
        else:
            clip = resize_with_scale_factor(clip=clip, scale_factor=tar_w_div_ori_w, interpolation_mode=self.interpolation_mode)
        # print('after resize', clip.shape)
        # print(clip.shape)
        # clip_tar_crop, i, j = random_crop(clip, self.size)
        clip_tar_crop, i, j = center_crop(clip, self.size)
        # print('after crop', clip_tar_crop.shape)

        return clip_tar_crop, ori_h, ori_w, i, j
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}"
    
class Compose:
    """Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.PILToTensor(),
        >>>     transforms.ConvertImageDtype(torch.float),
        >>> ])

    .. note::
        In order to script the transformations, please use ``torch.nn.Sequential`` as below.

        >>> transforms = torch.nn.Sequential(
        >>>     transforms.CenterCrop(10),
        >>>     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>> )
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.

    """

    def __init__(self, transforms):
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            _log_api_usage_once(self)
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            if isinstance(t, SDXLCenterCrop): # or isinstance(t, SDXL):
                img, ori_h, ori_w, crops_coords_top, crops_coords_left = t(img)
            else:
                img = t(img)
        return img, ori_h, ori_w, crops_coords_top, crops_coords_left

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string

video_transform =  Compose([
    ToTensorVideo(),
    # SDXLCenterCrop((self.height, 832)), # center crop using short edge, then resize
    SDXLCenterCrop((480, 832)), # center crop using short edge, then resize
    # video_transforms.SDXL((args.image_size[0], args.image_size[1])), # center crop using shor edge, then resize
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
])