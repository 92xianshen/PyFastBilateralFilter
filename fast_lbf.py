# height -> y, width -> x, depth -> z
# index: y, x, z

# 2021.09.24: `convn` implemented
# 2021.10.09: numpifying `splat` 

import numpy as np
from PIL import Image
import cv2
from skimage.transform import resize as skresize
import time

def clamp(min_value, max_value, x):
    return np.maximum(min_value, np.minimum(max_value, x))

def trilinear_interpolation(array, x, y, z):
    size_type, real_type = int, float
    # shape: y, x, z
    y_size, x_size, z_size = array.shape[:3]

    y_index = clamp(0, y_size - 1, size_type(y))
    yy_index = clamp(0, y_size - 1, y_index + 1)
    
    x_index = clamp(0, x_size - 1, size_type(x))
    xx_index = clamp(0, x_size - 1, x_index + 1)

    z_index = clamp(0, z_size - 1, size_type(z))
    zz_index = clamp(0, z_size - 1, z_index + 1)

    y_alpha = y - y_index
    x_alpha = x - x_index
    z_alpha = z - z_index

    return  (1. - x_alpha)  * (1. - y_alpha)    * (1. - z_alpha)    * array[y_index, x_index, z_index] + \
            x_alpha         * (1. - y_alpha)    * (1. - z_alpha)    * array[y_index, xx_index, z_index] + \
            (1. - x_alpha)  * y_alpha           * (1. - z_alpha)    * array[yy_index, x_index, z_index] + \
            x_alpha         * y_alpha           * (1. - z_alpha)    * array[yy_index, xx_index, z_index] + \
            (1. - x_alpha)  * (1. - y_alpha)    * z_alpha           * array[y_index, x_index, zz_index] + \
            x_alpha         * (1. - y_alpha)    * z_alpha           * array[y_index, xx_index, zz_index] + \
            (1. - x_alpha)  * y_alpha           * z_alpha           * array[yy_index, x_index, zz_index] + \
            x_alpha         * y_alpha           * z_alpha           * array[yy_index, xx_index, zz_index]


def trilinear_interpolation_np(array, y, x, z):
    # order: y, x, z
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

    yxz_index    = (y_index  * x_size * z_size + x_index  * z_size + z_index).reshape((-1)) # (h x w, )
    yxxz_index   = (y_index  * x_size * z_size + xx_index * z_size + z_index).reshape((-1)) # (h x w, )
    yyxz_index   = (yy_index * x_size * z_size + x_index  * z_size + z_index).reshape((-1)) # (h x w, )
    yyxxz_index  = (yy_index * x_size * z_size + xx_index * z_size + z_index).reshape((-1)) # (h x w, )
    yxzz_index   = (y_index  * x_size * z_size + x_index  * z_size + zz_index).reshape((-1)) # (h x w, )
    yxxzz_index  = (y_index  * x_size * z_size + xx_index * z_size + zz_index).reshape((-1)) # (h x w, )
    yyxzz_index  = (yy_index * x_size * z_size + x_index  * z_size + zz_index).reshape((-1)) # (h x w, )
    yyxxzz_index = (yy_index * x_size * z_size + xx_index * z_size + zz_index).reshape((-1)) # (h x w, )

    array = array.reshape((-1, 2)) # (h x w x c, 2) because it is `data`

    return  (1. - x_alpha) * (1. - y_alpha) * (1. - z_alpha) * array[yxz_index]   + \
            x_alpha        * (1. - y_alpha) * (1. - z_alpha) * array[yxxz_index]  + \
            (1. - x_alpha) * y_alpha        * (1. - z_alpha) * array[yyxz_index]  + \
            x_alpha        * y_alpha        * (1. - z_alpha) * array[yyxxz_index] + \
            (1. - x_alpha) * (1. - y_alpha) * z_alpha        * array[yxzz_index]  + \
            x_alpha        * (1. - y_alpha) * z_alpha        * array[yxxzz_index] + \
            (1. - x_alpha) * y_alpha        * z_alpha        * array[yyxzz_index] + \
            x_alpha        * y_alpha        * z_alpha        * array[yyxxzz_index]


def convn(data, buffer, n_iter):
    for _ in range(n_iter):
        buffer, data = data, buffer

        # For Dim y
        data[1:-1, 1:-1, 1:-1] = (buffer[:-2, 1:-1, 1:-1] + buffer[2:, 1:-1, 1:-1] + 2. * buffer[1:-1, 1:-1, 1:-1]) / 4.

        # For Dim x
        data[1:-1, 1:-1, 1:-1] = (buffer[1:-1, :-2, 1:-1] + buffer[1:-1, 2:, 1:-1] + 2. * buffer[1:-1, 1:-1, 1:-1]) / 4.

        # For Dim z
        data[1:-1, 1:-1, 1:-1] = (buffer[1:-1, 1:-1, :-2] + buffer[1:-1, 1:-1, 2:] + 2. * buffer[1:-1, 1:-1, 1:-1]) / 4.

def fast_LBF(inp, base, space_sigma, range_sigma, early_division, weight, result):
    size_type, real_type = int, float
    # height -> y, width -> x, depth -> z
    # index: y, x, z
    height, width = inp.shape[:2]
    padding_xy, padding_z = 2, 2
    base_min, base_max = base.min(), base.max()
    base_delta = base_max - base_min

    small_height = size_type((height - 1) / space_sigma) + 1 + 2 * padding_xy
    small_width = size_type((width - 1) / space_sigma) + 1 + 2 * padding_xy
    small_depth = size_type(base_delta / range_sigma) + 1 + 2 * padding_z

    print(small_height, small_width, small_depth)

    data = np.zeros((small_height, small_width, small_depth, 2), dtype='float32')

    # Splat
    # start = time.time()
    # for y in range(height):
    #     small_y = size_type(y / space_sigma + .5) + padding_xy
    #     for x in range(width):
    #         z = base[y, x] - base_min

    #         small_x = size_type(x / space_sigma + .5) + padding_xy
    #         small_z = size_type(z / range_sigma + .5) + padding_z

    #         d = data[small_y, small_x, small_z]
    #         d[0] += inp[y, x]
    #         d[1] += 1.
    # print('Time of splat with loop:', time.time() - start)
    # ->> Numpifying splat
    # ->> np.bincount
    print('Splatting...')
    data = data.reshape((-1, 2)) # flat, (small_h x small_w x small_depth, 2)
    inp = inp.reshape((-1, )) # flat, (h x w)
    # Space coordinates
    yy, xx = np.mgrid[:height, :width]
    small_yy = (yy / space_sigma + .5).astype(np.int32) + padding_xy # (h, w)
    small_xx = (xx / space_sigma + .5).astype(np.int32) + padding_xy # (h, w)
    # Range coordinates
    zz = base - base_min # (h, w)
    small_zz = (zz / range_sigma + .5).astype(np.int32) + padding_z # (h, w)
    # Coordinates
    coords = (small_yy * small_width * small_depth + small_xx * small_depth + small_zz).reshape((-1)) # (h x w)
    # Splatting
    data[:, 0] = np.bincount(coords, minlength=data.shape[0], weights=inp)
    data[:, 1] = np.bincount(coords, minlength=data.shape[0])
    data = data.reshape((small_height, small_width, small_depth, 2))
    print('Splatted.')
    
    buffer = np.zeros((small_height, small_width, small_depth, 2), dtype='float32')

    # Blur
    print('Blurring...')
    convn(data, buffer, n_iter=2)
    print('Blurred.')

    result = result.reshape((height, width))

    if early_division:
        print('Slicing...')
        data[..., 0] = np.divide(data[..., 0], data[..., 1], out=np.ones_like(data[..., 0]), where=data[..., 1] != 0)

        # Slice
        for y in range(height):
            for x in range(width):
                z = base[y, x] - base_min
                D = trilinear_interpolation(data, real_type(x) / space_sigma + padding_xy, real_type(y) / space_sigma + padding_xy, z / range_sigma + padding_z)

                result[y, x] = D[0]

        print('Sliced.')

    else:
        print('Slicing...')
        weight = weight.reshape((height, width))

        # Slice
        # start = time.time()
        # for y in range(height):
        #     for x in range(width):
        #         z = base[y, x] - base_min
        #         D = trilinear_interpolation(data, real_type(x) / space_sigma + padding_xy, real_type(y) / space_sigma + padding_xy, z / range_sigma + padding_z)

        #         weight[y, x] = D[1]
        #         result[y, x] = D[0] / (D[1] + 1e-10)
        # print('Time of slice with loop:', time.time() - start)
        # ->> Numpifying
        yy, xx = np.mgrid[:height, :width] # (h, w)
        zz = base - base_min # (h, w)
        small_yy = yy.astype(np.float32) / space_sigma + padding_xy
        small_xx = xx.astype(np.float32) / space_sigma + padding_xy
        small_zz = zz / range_sigma + padding_z
        D = trilinear_interpolation_np(data, small_yy, small_xx, small_zz)
        weight[:] = D.reshape((height, width, 2))[..., 1]
        result[:] = D.reshape((height, width, 2))[..., 0] / (weight + 1e-10)
        print('Sliced.')

im = Image.open('lena.jpg').convert('L')
im = np.asarray(im, dtype='float32') / 255.

height, width = im.shape

weight = np.zeros_like(im)
result = np.zeros_like(im)

im_down = skresize(skresize(im, (height // 16, width // 16)), (height, width))

fast_LBF(im_down, im, space_sigma=16., range_sigma=.0625, early_division=False, weight=weight, result=result)

cv2.imshow('im', im)
cv2.imshow('down', im_down)
cv2.imshow('result', result)
cv2.waitKey()