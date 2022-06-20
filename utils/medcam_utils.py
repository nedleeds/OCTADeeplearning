import cv2
import numpy as np
import matplotlib.cm as cm
import nibabel as nib
import torch
from torch.nn import functional as F
from functools import reduce
import operator
from pathlib import Path
import os
from functools import wraps

MIN_SHAPE = (500, 500)

def check_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # print(func.__name__, end=' -> ')
        out = func(*args, **kwargs)
        # print('done')
        return out
    return wrapper
    
@check_call
def save_attention_map(filename, attention_map, heatmap, raw_input, nib_info=None):
    """
    Saves an attention maps.
    Args:
        filename: The save path, including the name, excluding the file extension.
        attention_map: The attention map in HxW or DxHxW format.
        heatmap: If the attention map should be saved as a heatmap. True for gcam and gcampp. False for gbp and ggcam.
    """
    # print(f'map shape : {attention_map.shape}')
    if isinstance(attention_map, torch.Tensor):
        attention_map = attention_map.detach().cpu().numpy()
    dim = len(attention_map.shape)
    
    if isinstance(raw_input, torch.Tensor):
        if dim == 3 and raw_input.shape[0]==1:
            raw_input = raw_input.squeeze_(0)
            raw_input = raw_input.detach().cpu().numpy()
        if dim ==2 and raw_input.shape[0] == 1 or raw_input.shape[0] == 3:
            raw_input = raw_input.detach().cpu().numpy()
            raw_input = raw_input.transpose(2, 1, 0)
            
    if raw_input.max() != 1 or raw_input.min()!=0:
        raw_input = normalize(raw_input.astype(np.float))
        # print('raw_input has been min-max nmzed.')
    attention_map = normalize(attention_map.astype(np.float))
    attention_map = generate_attention_map(attention_map, heatmap, dim, raw_input)
    
    # print('generated att map size :', attention_map.shape)
    
    if dim==3:
        interpolated_map = _save_file(filename, attention_map, dim, nib_info, shape=(raw_input.shape))
        _save_file(filename + '_raw', raw_input, dim, nib_info)
        overlay = make_enface_and_overlay(raw_input, interpolated_map)
        if int(filename.split('/')[-1].split('_')[0]) > 10300:
            overlay = np.rot90(overlay, 1)
        else:
            overlay = np.fliplr(overlay)
        overlay = np.rot90(overlay, 1)
        overlay = np.flipud(overlay)
        _save_file(filename + '_overlay', overlay, dim=2)
    else:
        attention_map = np.rot90(attention_map, 1)
        attention_map = np.flipud(attention_map)
        _save_file(filename, attention_map, dim, nib_info, shape=(raw_input.shape))

@check_call
def generate_attention_map(attention_map, heatmap, dim, raw_input):
    if dim == 2:
        if heatmap:
            return generate_gcam2d(attention_map, raw_input)
        else:
            return generate_guided_bp2d(attention_map)
    elif dim == 3:
        if heatmap:
            return generate_gcam3d(attention_map)
        else:
            return generate_guided_bp3d(attention_map)
    else:
        raise RuntimeError("Unsupported dimension. Only 2D and 3D data is supported.")

@check_call
def generate_gcam2d(attention_map, raw_input):
    assert(len(attention_map.shape) == 2)  # No batch dim
    assert(isinstance(attention_map, np.ndarray))  # Not a tensor

    if raw_input is not None:
        attention_map = overlay(raw_input, attention_map)
    else:
        attention_map = _resize_attention_map(attention_map, MIN_SHAPE)
        attention_map = cm.jet_r(attention_map)[..., :3] * 255.0
    return np.uint8(attention_map)

@check_call
def generate_guided_bp2d(attention_map):
    assert(len(attention_map.shape) == 2)
    assert (isinstance(attention_map, np.ndarray))  # Not a tensor

    attention_map *= 255.0
    attention_map = _resize_attention_map(attention_map, MIN_SHAPE)
    return np.uint8(attention_map)

@check_call
def generate_gcam3d(attention_map, data=None):
    assert(isinstance(attention_map, np.ndarray))  # Not a tensor
    assert(isinstance(data, np.ndarray) or data is None)  # Not PIL
    assert(data is None or len(data.shape) == 3)
    
    attention_map *= 255.0
    return np.uint8(attention_map)

@check_call
def generate_guided_bp3d(attention_map):
    assert(len(attention_map.shape) == 3)
    assert (isinstance(attention_map, np.ndarray))  # Not a tensor

    attention_map *= 255.0
    return np.uint8(attention_map)

@check_call
def _load_data(data_path):
    if isinstance(data_path, str):
        return cv2.imread(data_path)
    else:
        return data_path

@check_call
def _resize_attention_map(attention_map, min_shape):
    attention_map_shape = attention_map.shape[:2]
    if min(min_shape) < min(attention_map_shape):
        attention_map = cv2.resize(attention_map, tuple(np.flip(attention_map_shape)))
    else:
        resize_factor = int(min(min_shape) / min(attention_map_shape))
        data_shape = (attention_map_shape[0] * resize_factor, attention_map_shape[1] * resize_factor)
        attention_map = cv2.resize(attention_map, tuple(np.flip(data_shape)))
    return attention_map

@check_call
def normalize(x):
    """Normalizes data both numpy or tensor data to range [0,1]."""
    if isinstance(x, torch.Tensor):
        if torch.min(x) == torch.max(x):
            return torch.zeros(x.shape)
        return (x-torch.min(x))/(torch.max(x)-torch.min(x))
    else:
        if np.min(x) == np.max(x):
            return np.zeros(x.shape)
        return (x - np.min(x)) / (np.max(x) - np.min(x))

@check_call
def _save_file(filename, attention_map, dim, nib_info=None, shape=None):
    Path(os.path.dirname(filename)).mkdir(parents=True, exist_ok=True)
    nii_name = 'raw' if 'raw' in filename else 'att'
        
    if dim == 2:
        cv2.imwrite(filename + ".png", attention_map)
        return None
    else:
        if shape is not None:
            attention_map = interpolate(attention_map, shape=shape, squeeze=True)
            # print(f'interpolated size : {attention_map.shape}')
        if np.argmax(attention_map.shape) == 0:
            attention_map = attention_map.transpose(1, 0, 2)
    
        # customized
        if nii_name == 'att': 
            mask_dir = './data/Nifti/In/Transformed/Mask_SRL_256'
            mask_name = filename.split('/')[-1].split('_')[0]+'.nii.gz'
            mask_path = os.path.join(mask_dir, mask_name)
            attention_map = masking(mask_path , attention_map)
            attention_map_arr = attention_map
        
        attention_map = nib.Nifti1Image(attention_map, affine=nib_info['affine'], header=nib_info['header'])
        nib.save(attention_map, filename + ".nii.gz")
        if ('raw' not in filename):
            # print('save att map')
            return attention_map_arr
        else:
            # print('save raw')
            return None
        

def masking(mask_path, attention_map):
    mask = np.asarray(nib.load(mask_path).dataobj)
    masked_map = np.zeros(np.shape(attention_map))
    masked_map[mask>0] = attention_map[mask>0]
    return masked_map

@check_call
def get_layers(model, reverse=False):
    """Returns the layers of the model. Optionally reverses the order of the layers."""
    layer_names = []
    for name, _ in model.named_modules():
        layer_names.append(name)

    if layer_names[0] == "":
        layer_names = layer_names[1:]

    index = 0
    sub_index = 0
    while True:
        if index == len(layer_names) - 1:
            break
        if sub_index < len(layer_names) - 1 and layer_names[index] == layer_names[sub_index + 1][:len(layer_names[index])]:
            sub_index += 1
        elif sub_index > index:
            layer_names.insert(sub_index, layer_names.pop(index))
            sub_index = index
        else:
            index += 1
            sub_index = index

    if reverse:
        layer_names.reverse()

    return layer_names

@check_call
def interpolate(data, shape, squeeze=False):
    """Interpolates data to the size of a given shape. Optionally squeezes away the batch and channel dim if the data was given in HxW or DxHxW format.
    Shape needs to be as size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int])"""
    if isinstance(data, np.ndarray):
        # Lazy solution, numpy and scipy have multiple interpolate methods with only linear or nearest, so I don't know which one to use... + they don't work with batches
        # Should be redone with numpy or scipy though
        data_type = data.dtype
        data = torch.FloatTensor(data)
        data = _interpolate_tensor(data, shape, squeeze)
        data = data.numpy().astype(data_type)
    elif isinstance(data, torch.Tensor):
        data = _interpolate_tensor(data, shape, squeeze)
    else:
        raise ValueError("Unsupported data type for interpolation")
    return data

@check_call
def _interpolate_tensor(data, shape, squeeze):
    """Interpolates data to the size of a given shape. Optionally squeezes away the batch and channel dim if the data was given in HxW or DxHxW format.
    Shape needs to be as size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int])"""
    _squeeze = 0
    if (len(shape) == 2 and len(data.shape) == 2) or ((len(shape) == 3 and len(data.shape) == 3)):  # Add batch and channel dim
        data = data.unsqueeze(0).unsqueeze(0)
        _squeeze = 2
    elif (len(shape) == 2 and len(data.shape) == 3) or ((len(shape) == 3 and len(data.shape) == 4)):  # Add batch dim
        data = data.unsqueeze(0)
        _squeeze = 1

        
    if len(shape) == 2:
        data = F.interpolate(data, shape, mode="bilinear", align_corners=False)
    else:
        data = F.interpolate(data, shape, mode="trilinear", align_corners=False)
    if squeeze:  # Remove unnecessary dims
        for i in range(_squeeze):
            data = data.squeeze(0)
    return data

@check_call
def prod(iterable):
    return reduce(operator.mul, iterable, 1)

@check_call
def overlay(raw_input, attention_map):
    if np.max(raw_input) > 1:
        raw_input = raw_input.astype(np.float)
        raw_input /= 255
    #(192,192,1)/(7,7) : raw_input size, attention_map size
    attention_map = cv2.resize(attention_map, tuple(np.flip(raw_input.shape[:2]))) #(192,192,1)/(192,192)
    attention_map = cm.jet_r(attention_map)[..., :3] #(192,192,1)/(192,192,3)
    attention_map = (attention_map.astype(np.float) + raw_input.astype(np.float)) / 2
    attention_map *= 255
    return attention_map

@check_call
def make_enface(nii):
    # print('before enface :',np.shape(nii))
    if np.argmax(np.shape(nii)) == 1:
        nii = nii.transpose(1, 0, 2)
    z_axis = np.argmax(np.shape(nii))
    enface = np.max(nii, axis=z_axis)
    # print('after enface :', np.shape(enface))
    return enface

@check_call
def make_enface_and_overlay(raw_input, attention_map):
    assert not isinstance(raw_input, torch.Tensor), "raw array needed for make_enface"
    assert not isinstance(attention_map, torch.Tensor), "attention array needed for make_enface"
    # input should be array. not a tensor
    raw_enface = make_enface(raw_input)
    attention_enface = make_enface(attention_map)
    # print(f'raw/att enface shape: {raw_enface.shape}/{attention_enface.shape}')
    if raw_enface.max() != 1 or raw_enface.min() != 0 :
        raw_enface = normalize(raw_enface)
    elif attention_enface.max() != 1 or attention_enface.min() != 0:
        attention_enface = normalize(attention_enface)
    raw_enface = raw_enface[:, :, np.newaxis]
    attention_enface = cm.jet_r(attention_enface)[..., :3]
    # print(f'raw/att enface shape: {raw_enface.shape}/{attention_enface.shape}')
    overlay = (attention_enface.astype(np.float) + raw_enface.astype(np.float)) / 2
    overlay *= 255
    return overlay
    

@check_call
def unpack_tensors_with_gradients(tensors):
    unpacked_tensors = []
    if isinstance(tensors, torch.Tensor):
        if tensors.requires_grad:
            return [tensors]
        else:
            return []
    elif isinstance(tensors, dict):
        for value in tensors.values():
            unpacked_tensors.extend(unpack_tensors_with_gradients(value))
        return unpacked_tensors
    elif isinstance(tensors, list):
        for value in tensors:
            unpacked_tensors.extend(unpack_tensors_with_gradients(value))
        return unpacked_tensors
    else:
        raise ValueError("Cannot unpack unknown data type.")
