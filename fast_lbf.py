# height -> y, width -> x, depth -> z
# index: y, x, z

# 2021.09.24: `convn` implemented
# 2021.10.09: Numpifying `splat` 
# 2021.10.09: `splat` numpified
# 2021.10.09: Numpifying `slice`
# 2021.10.09: `slice` numpified

import numpy as np
from PIL import Image
import cv2
from numpy.core.fromnumeric import repeat
from skimage.transform import resize as skresize
import itertools as it

def clamp(min_value, max_value, x):
    return np.maximum(min_value, np.minimum(max_value, x))

def trilinear_interpolation(array, y, x, z):
    # Index order: y, x, z
    y_size, x_size, z_size = array.shape[:3]

    y_index  = clamp(0, y_size - 1, y.astype(np.int32)) # (h, w)
    yy_index = clamp(0, y_size - 1, y_index + 1) # (h, w)

    x_index  = clamp(0, x_size - 1, x.astype(np.int32)) # (h, w)
    xx_index = clamp(0, x_size - 1, x_index + 1) # (h, w)

    z_index  = clamp(0, z_size - 1, z.astype(np.int32)) # (h, w)
    zz_index = clamp(0, z_size - 1, z_index + 1) # (h, w)

    y_alpha = (y - y_index).reshape((-1, 1)) # (h x w, )
    x_alpha = (x - x_index).reshape((-1, 1)) # (h x w, )
    z_alpha = (z - z_index).reshape((-1, 1)) # (h x w, )

    # Coordinates
    yxz_index    = (y_index  * x_size * z_size + x_index  * z_size + z_index).reshape((-1)) # (h x w, )
    yxxz_index   = (y_index  * x_size * z_size + xx_index * z_size + z_index).reshape((-1)) # (h x w, )
    yyxz_index   = (yy_index * x_size * z_size + x_index  * z_size + z_index).reshape((-1)) # (h x w, )
    yyxxz_index  = (yy_index * x_size * z_size + xx_index * z_size + z_index).reshape((-1)) # (h x w, )
    yxzz_index   = (y_index  * x_size * z_size + x_index  * z_size + zz_index).reshape((-1)) # (h x w, )
    yxxzz_index  = (y_index  * x_size * z_size + xx_index * z_size + zz_index).reshape((-1)) # (h x w, )
    yyxzz_index  = (yy_index * x_size * z_size + x_index  * z_size + zz_index).reshape((-1)) # (h x w, )
    yyxxzz_index = (yy_index * x_size * z_size + xx_index * z_size + zz_index).reshape((-1)) # (h x w, )

    # Shape (h x w x c, 2) because it is `data`, idx 0 is input, idx 1 is weight
    array = array.reshape((-1, 2))

    return  (1. - x_alpha) * (1. - y_alpha) * (1. - z_alpha) * array[yxz_index]   + \
            x_alpha        * (1. - y_alpha) * (1. - z_alpha) * array[yxxz_index]  + \
            (1. - x_alpha) * y_alpha        * (1. - z_alpha) * array[yyxz_index]  + \
            x_alpha        * y_alpha        * (1. - z_alpha) * array[yyxxz_index] + \
            (1. - x_alpha) * (1. - y_alpha) * z_alpha        * array[yxzz_index]  + \
            x_alpha        * (1. - y_alpha) * z_alpha        * array[yxxzz_index] + \
            (1. - x_alpha) * y_alpha        * z_alpha        * array[yyxzz_index] + \
            x_alpha        * y_alpha        * z_alpha        * array[yyxxzz_index]

def loop_trilinear_interpolation(array, y, x, z, height, width):
    # Index order: y, x, z
    y_size, x_size, z_size = array.shape[:3]

    def bilateral_coord_transform(y_idx, x_idx, z_idx):
        return np.reshape((y_idx * x_size + x_idx) * z_size + z_idx, (-1, ))

    coord_transform = bilateral_coord_transform

    # Method to get left and right indices of slice interpolation
    def get_both_indices(size, coord):
        left_index = clamp(0, size - 1, coord.astype(np.int32))
        right_index = clamp(0, size - 1, left_index + 1)
        return left_index, right_index

    # Spatial interpolation index of slice
    y_index, yy_index = get_both_indices(y_size, y) # (h, w)
    x_index, xx_index = get_both_indices(x_size, x) # (h, w)
    z_index, zz_index = get_both_indices(z_size, z)

    # Spatial interpolation factor of slice
    y_alpha = np.reshape(y - y_index, [-1, ]) # (h x w)
    x_alpha = np.reshape(x - x_index, [-1, ]) # (h x w)
    z_alpha = np.reshape(z - z_index, [-1, ])

    interp_indices = np.asarray([y_index, yy_index, x_index, xx_index, z_index, zz_index]) # (10, h x w)
    alphas = np.asarray([1. - y_alpha, y_alpha, 1. - x_alpha, x_alpha, 1. - z_alpha, z_alpha]) # (10, h x w)

    interpolation = np.zeros((height, width, 2), dtype=np.float32).reshape((-1, 2))
    offset = np.arange(3, dtype=np.int32) * 2
    array_flat = array.reshape((-1, 2))

    for perm in it.product(range(2), repeat=3):
        print(perm, np.asarray(perm), np.asarray(perm) + offset)
        alpha_prod = alphas[np.asarray(perm) + offset]
        idx = interp_indices[np.asarray(perm) + offset]
        print(coord_transform(*idx))

        data_slice = array_flat[coord_transform(*idx)]

        interpolation += np.prod(alpha_prod, axis=0)[..., np.newaxis] * data_slice

    return interpolation


def convn0(data, buffer, n_iter):
    for _ in range(n_iter):
        buffer, data = data, buffer

        # For Dim y
        data[1:-1, 1:-1, 1:-1] = (buffer[:-2, 1:-1, 1:-1] + buffer[2:, 1:-1, 1:-1] + 2. * buffer[1:-1, 1:-1, 1:-1]) / 4.

        # For Dim x
        data[1:-1, 1:-1, 1:-1] = (buffer[1:-1, :-2, 1:-1] + buffer[1:-1, 2:, 1:-1] + 2. * buffer[1:-1, 1:-1, 1:-1]) / 4.

        # For Dim z
        data[1:-1, 1:-1, 1:-1] = (buffer[1:-1, 1:-1, :-2] + buffer[1:-1, 1:-1, 2:] + 2. * buffer[1:-1, 1:-1, 1:-1]) / 4.

def convn(data, buffer, n_iter, n_dim):
    perm = list(range(1, n_dim - 1)) + [0, n_dim - 1] # [1, ..., ndim - 2, 0, ndim - 1] because last dim is 2, comprising domain and weight
    
    for _ in range(n_iter):
        buffer, data = data, buffer

        for dim in range(n_dim - 1):
            data[1:-1] = (buffer[:-2] + buffer[2:] + 2. * buffer[1:-1]) / 4.
            data = np.transpose(data, perm)
            buffer = np.transpose(buffer, perm)

def fast_LBF(inp, base, space_sigma, range_sigma, early_division, weight, result):
    # Datatype cast
    size_type, real_type = int, float
    
    # Index order: y --> height, x --> width, z --> depth
    height, width = inp.shape[:2]
    padding_xy, padding_z = 2, 2
    base_min, base_max = base.min(), base.max()
    base_delta = base_max - base_min

    # Space coordinates, shape (h, w)
    yy, xx = np.mgrid[:height, :width]
    # Range coordinates, shape (h, w)
    zz = base - base_min

    # Data shape
    small_height = size_type((height - 1) / space_sigma) + 1 + 2 * padding_xy
    small_width = size_type((width - 1) / space_sigma) + 1 + 2 * padding_xy
    small_depth = size_type(base_delta / range_sigma) + 1 + 2 * padding_z

    data = np.zeros((small_height, small_width, small_depth, 2), dtype='float32')

    # ==== Splat ====
    # ->> Numpifying
    print('Splatting...')
    
    # Flatten, shape (small_height x small_width x small_depth, 2)
    data = data.reshape((-1, 2))
    # Flatten, shape (h x w)
    inp = inp.reshape((-1, )) 
    
    # Space coordinates, shape (h, w)
    small_yy = (yy / space_sigma + .5).astype(np.int32) + padding_xy
    small_xx = (xx / space_sigma + .5).astype(np.int32) + padding_xy
    # Range coordinates, shape (h, w)
    small_zz = (zz / range_sigma + .5).astype(np.int32) + padding_z
    
    # Coordinates, shape (h x w, )
    coords = (small_yy * small_width * small_depth + small_xx * small_depth + small_zz).reshape((-1))
    
    # Splatting
    data[:, 0] = np.bincount(coords, minlength=data.shape[0], weights=inp)
    data[:, 1] = np.bincount(coords, minlength=data.shape[0])
    
    # Reshape
    data = data.reshape((small_height, small_width, small_depth, 2))
    print('Splatted.')
    
    # ==== Blur ====
    print('Blurring...')
    buffer = np.zeros((small_height, small_width, small_depth, 2), dtype='float32')
    # 3D convolution
    n_dim = data.ndim
    convn(data, buffer, n_iter=2, n_dim=n_dim)
    print('Blurred.')

    result = result.reshape((height, width))

    if early_division:
        # ==== Slice ====
        # ->> Numpifying
        print('Slicing...')
        
        # Early division
        data[..., 0] = np.divide(
            data[..., 0], data[..., 1], out=np.ones_like(data[..., 0]), where=data[..., 1] != 0)
        
        # Space coordinates, shape (h, w)
        small_yy = yy.astype(np.float32) / space_sigma + padding_xy
        small_xx = xx.astype(np.float32) / space_sigma + padding_xy
        # Range coordinates, shape (h, w)
        small_zz = zz / range_sigma + padding_z
        
        # Interpolation
        D = trilinear_interpolation(data, small_yy, small_xx, small_zz)

        # Get result
        result[:] = D.reshape((height, width, 2))[..., 0]
        
        print('Sliced.')

    else:
        # ==== Slice ====
        # ->> Numpifying
        print('Slicing...')
        
        weight = weight.reshape((height, width))
        
        # Space coordinates, shape (h, w)
        small_yy = yy.astype(np.float32) / space_sigma + padding_xy
        small_xx = xx.astype(np.float32) / space_sigma + padding_xy
        # Range coordinates, shape (h, w)
        small_zz = zz / range_sigma + padding_z
        
        # Interpolation
        D = trilinear_interpolation(data, small_yy, small_xx, small_zz)
        D2 = loop_trilinear_interpolation(data, small_yy, small_xx, small_zz, height, width)

        print('trilinear interpolation is close or not?', np.allclose(D, D2))

        # Get weight and result
        weight[:] = D2.reshape((height, width, 2))[..., 1]
        result[:] = D2.reshape((height, width, 2))[..., 0] / (weight + 1e-10)
        
        print('Sliced.')

im = Image.open('lena.jpg').convert('L')
im = np.asarray(im, dtype='float32') / 255.

height, width = im.shape

weight = np.zeros_like(im)
result = np.zeros_like(im)

fast_LBF(im, im, space_sigma=16., range_sigma=.0625, early_division=False, weight=weight, result=result)

cv2.imshow('im', im)
cv2.imshow('result', result)
cv2.waitKey()