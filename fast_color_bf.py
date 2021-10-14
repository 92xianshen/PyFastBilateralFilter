# height -> y, width -> x, depth -> z
# index: y, x, z

# 2021.09.24: `convn` implemented
# 2021.10.09: Numpifying `splat` 
# 2021.10.09: `splat` numpified
# 2021.10.09: Numpifying `slice`
# 2021.10.09: `slice` numpified
# 2021.10.12: 3-channel (r, g, and b) bilateral filter created

import numpy as np
from PIL import Image
import cv2
from skimage.transform import resize as skresize
import itertools as it

def clamp(min_value, max_value, x):
    return np.maximum(min_value, np.minimum(max_value, x))

def loop_Nlinear_interpolation(array, y, x, r, g, b):
    # Index order: y, x, r, g, b
    y_size, x_size, r_size, g_size, b_size = array.shape

    # Flatten
    array = array.reshape(-1)

    y_index  = clamp(0, y_size - 1, y.astype(np.int32)) # (h, w)
    yy_index = clamp(0, y_size - 1, y_index + 1) # (h, w)

    x_index  = clamp(0, x_size - 1, x.astype(np.int32)) # (h, w)
    xx_index = clamp(0, x_size - 1, x_index + 1) # (h, w)

    r_index  = clamp(0, r_size - 1, r.astype(np.int32)) # (h, w)
    rr_index = clamp(0, r_size - 1, r_index + 1) # (h, w)

    g_index  = clamp(0, g_size - 1, g.astype(np.int32)) # (h, w)
    gg_index = clamp(0, g_size - 1, g_index + 1) # (h, w)

    b_index  = clamp(0, b_size - 1, b.astype(np.int32)) # (h, w)
    bb_index = clamp(0, b_size - 1, b_index + 1) # (h, w)

    left_indices = [y_index, x_index, r_index, g_index, b_index]
    right_indices = [yy_index, xx_index, rr_index, gg_index, bb_index]

    # Coordinates
    def coord_transform(y_idx, x_idx, r_idx, g_idx, b_idx):
        return ((((y_idx * x_size + x_idx) * r_size + r_idx) * g_size + g_idx) * b_size + b_idx).reshape((-1))

    y_alpha = (y - y_index).reshape((-1, )) # (h x w, )
    x_alpha = (x - x_index).reshape((-1, )) # (h x w, )
    r_alpha = (r - r_index).reshape((-1, )) # (h x w, )
    g_alpha = (g - g_index).reshape((-1, )) # (h x w, )
    b_alpha = (b - b_index).reshape((-1, )) # (h x w, )
    alphas = [y_alpha, x_alpha, r_alpha, g_alpha, b_alpha] # (5, h x w)

    interpolation = np.zeros_like(y_alpha)
    for perm in it.product(range(2), repeat=5): # (0, 0, 0, 0, 1) for example
        alpha_prod = np.ones_like(y_alpha)
        idx = []
        for i in range(len(perm)):
            if perm[i] == 1:
                alpha_prod *= (1. - alphas[i]) # (h x w, )
                idx.append(left_indices[i]) 
            elif perm[i] == 0:
                alpha_prod *= alphas[i]
                idx.append(right_indices[i])
            else:
                raise ValueError

        interpolation += alpha_prod * array[coord_transform(*idx)]

    return interpolation

def convn(data, buffer, n_iter):
    perm = list(range(1, data.ndim)) + [0] # [1, ..., ndim - 1, 0] 
    
    for _ in range(n_iter):
        buffer, data = data, buffer

        for dim in range(data.ndim):
            data[1:-1] = (buffer[:-2] + buffer[2:] + 2. * buffer[1:-1]) / 4.
            data = np.transpose(data, perm)
            buffer = np.transpose(buffer, perm)

def fast_color_BF(inp, base, space_sigma, range_sigma, result, weight):
    '''
    Fast color bilateral filter

    Args:
        inp: input image to be filtered, shape (h, w, c_1)
        base: edge or guidance image, shape (h, w, 3)
        space_sigma: sigma_s, float
        range_sigma: sigma_r, float
        weight: returned weight array, shape (h, w, c_1)
        result: returned result array, shape (h, w, c_1)

    Returns:
        None, because `weight` and `result` are referenced 
    '''
    # Datatype cast
    size_type, real_type = int, float

    # Index order: y --> height, x --> width, z --> depth
    height, width, n_channels1 = inp.shape # (h, w, c_1)
    padding_xy, padding_z = 2, 2

    # base comprises r, g, b channels
    r, g, b = base[..., 0], base[..., 1], base[..., 2]
    r_min, r_max = r.min(), r.max()
    g_min, g_max = g.min(), g.max()
    b_min, b_max = b.min(), b.max()
    r_delta, g_delta, b_delta = r_max - r_min, g_max - g_min, b_max - b_min

    # Space coordinates, shape (h, w), dtype int
    yy, xx = np.mgrid[:height, :width]
    # Range coordinates, shape (h, w, 3), dtype float
    rr, gg, bb = r - r_min, g - g_min, b - b_min

    # Shape of `data`, scala, dtype size_type
    small_height = size_type((height - 1) / space_sigma) + 1 + 2 * padding_xy 
    small_width = size_type((width - 1) / space_sigma) + 1 + 2 * padding_xy
    small_rdepth = size_type(r_delta / range_sigma) + 1 + 2 * padding_z 
    small_gdepth = size_type(g_delta / range_sigma) + 1 + 2 * padding_z 
    small_bdepth = size_type(b_delta / range_sigma) + 1 + 2 * padding_z

    # Space coordinates, shape (h, w)
    splat_yy = (yy / space_sigma + .5).astype(np.int32) + padding_xy
    splat_xx = (xx / space_sigma + .5).astype(np.int32) + padding_xy
    # Range coordinates, shape (h, w)
    splat_rr = (rr / range_sigma + .5).astype(np.int32) + padding_z
    splat_gg = (gg / range_sigma + .5).astype(np.int32) + padding_z
    splat_bb = (bb / range_sigma + .5).astype(np.int32) + padding_z
    # Splat coordinates, shape (h x w, )
    splat_coords = (((splat_yy * small_width + splat_xx) * small_rdepth + splat_rr) * small_gdepth + splat_gg) * small_bdepth + splat_bb
    splat_coords = splat_coords.reshape((-1)) # (h x w, )

    # Slice coordinates
    # Space coordinates, shape (h, w)
    slice_yy = yy.astype(np.float32) / space_sigma + padding_xy
    slice_xx = xx.astype(np.float32) / space_sigma + padding_xy
    # Range coordinates, shape (h, w)
    slice_rr = rr / range_sigma + padding_z
    slice_gg = gg / range_sigma + padding_z
    slice_bb = bb / range_sigma + padding_z

    # High-dim bilateral grid
    data = np.zeros((small_height, small_width, small_rdepth, small_gdepth, small_bdepth), dtype=np.float32) # For each channel

    # For each channel
    for ch in range(n_channels1 + 1):
        if ch == n_channels1:
            print('Processing weight')
            inp0 = np.ones((height, width), dtype=np.float32)
            result0 = weight[..., 0]
        else:
            print('Processing channel', ch)
            inp0 = inp[..., ch]
            result0 = result[..., ch]
        # ==== Splat ====
        print('Splatting...')

        # Flatten and reset, shape (small_height x small_width x small_depth1 x ... x small_depthc, )
        data = data.reshape((-1)) * 0
        # Flatten, shape (h x w)
        inp0 = inp0.reshape((-1))

        # Splatting
        data[:] = np.bincount(splat_coords, minlength=data.shape[0], weights=inp0)

        # Reshape
        data = data.reshape((small_height, small_width, small_rdepth, small_gdepth, small_bdepth))
        print('Splatted.')

        # ==== Blur ====
        print('Blurring...')
        buffer = np.zeros_like(data)
        # ND convolution, maybe 5D
        convn(data, buffer, n_iter=2)
        print('Blurred.')

        # ==== Slice ====
        print('Slicing...')

        # Interpolation
        D = loop_Nlinear_interpolation(data, slice_yy, slice_xx, slice_rr, slice_gg, slice_bb)

        # Get result0
        result0[:] = D.reshape((height, width))

        print('Sliced.')
        
im = Image.open('lena.jpg')
im = np.asarray(im, dtype='float32') / 255.

height, width, n_channels = im.shape
inp = im[..., 0:1]

result = np.zeros_like(inp)
weight = np.zeros_like(inp)

fast_color_BF(inp, im, space_sigma=16., range_sigma=.25, result=result, weight=weight)

result = result / (weight + 1e-5)

cv2.imshow('im', im[..., ::-1])
cv2.imshow('result', result[..., 0])
cv2.waitKey()