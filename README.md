PyFastBilateralFilter
=====================

A python implementation of fast bialteral filter

fast_lbf.py
-----------

Grayscale bilateral filter

fast_color_bf.py
----------------

Multi-channel bilateral filter

- Find mins and maxs for each channel
- `data` is five-dimensional (2 spatial + 3 range coordinates)
- 2021.10.13 `fast_color_BF` with three-channel base created