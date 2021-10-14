# def naive_Nlinear_interpolation(array, y, x, r, g, b):
#     # Index order: y, x, r, g, b
#     y_size, x_size, r_size, g_size, b_size = array.shape

#     y_index  = clamp(0, y_size - 1, y.astype(np.int32)) # (h, w)
#     yy_index = clamp(0, y_size - 1, y_index + 1) # (h, w)

#     x_index  = clamp(0, x_size - 1, x.astype(np.int32)) # (h, w)
#     xx_index = clamp(0, x_size - 1, x_index + 1) # (h, w)

#     r_index  = clamp(0, r_size - 1, r.astype(np.int32)) # (h, w)
#     rr_index = clamp(0, r_size - 1, r_index + 1) # (h, w)

#     g_index  = clamp(0, g_size - 1, g.astype(np.int32)) # (h, w)
#     gg_index = clamp(0, g_size - 1, g_index + 1) # (h, w)

#     b_index  = clamp(0, b_size - 1, b.astype(np.int32)) # (h, w)
#     bb_index = clamp(0, b_size - 1, b_index + 1) # (h, w)

#     y_alpha = (y - y_index).reshape((-1, )) # (h x w, )
#     x_alpha = (x - x_index).reshape((-1, )) # (h x w, )
#     r_alpha = (r - r_index).reshape((-1, )) # (h x w, )
#     g_alpha = (g - g_index).reshape((-1, )) # (h x w, )
#     b_alpha = (b - b_index).reshape((-1, )) # (h x w, )

#     # Coordinates
#     def coord_transform(y_idx, x_idx, r_idx, g_idx, b_idx):
#         return ((((y_idx * x_size + x_idx) * r_size + r_idx) * g_size + g_idx) * b_size + b_idx).reshape((-1))
    
#     # (h x w, )
#     yxrgb_idx = coord_transform(y_index, x_index, r_index, g_index, b_index) 
#     yxxrgb_idx = coord_transform(y_index, xx_index, r_index, g_index, b_index)
#     yyxrgb_idx = coord_transform(yy_index, x_index, r_index, g_index, b_index) 
#     yyxxrgb_idx = coord_transform(yy_index, xx_index, r_index, g_index, b_index)
#     yxrrgb_idx = coord_transform(y_index, x_index, rr_index, g_index, b_index) 
#     yxxrrgb_idx = coord_transform(y_index, xx_index, rr_index, g_index, b_index) 
#     yyxrrgb_idx = coord_transform(yy_index, x_index, rr_index, g_index, b_index) 
#     yyxxrrgb_idx = coord_transform(yy_index, xx_index, rr_index, g_index, b_index) 
#     yxrggb_idx = coord_transform(y_index, x_index, r_index, gg_index, b_index) 
#     yxxrggb_idx = coord_transform(y_index, xx_index, r_index, gg_index, b_index)
#     yyxrggb_idx = coord_transform(yy_index, x_index, r_index, gg_index, b_index) 
#     yyxxrggb_idx = coord_transform(yy_index, xx_index, r_index, gg_index, b_index)
#     yxrrggb_idx = coord_transform(y_index, x_index, rr_index, gg_index, b_index) 
#     yxxrrggb_idx = coord_transform(y_index, xx_index, rr_index, gg_index, b_index) 
#     yyxrrggb_idx = coord_transform(yy_index, x_index, rr_index, gg_index, b_index) 
#     yyxxrrggb_idx = coord_transform(yy_index, xx_index, rr_index, gg_index, b_index) 
#     yxrgbb_idx = coord_transform(y_index, x_index, r_index, g_index, bb_index) 
#     yxxrgbb_idx = coord_transform(y_index, xx_index, r_index, g_index, bb_index)
#     yyxrgbb_idx = coord_transform(yy_index, x_index, r_index, g_index, bb_index) 
#     yyxxrgbb_idx = coord_transform(yy_index, xx_index, r_index, g_index, bb_index)
#     yxrrgbb_idx = coord_transform(y_index, x_index, rr_index, g_index, bb_index) 
#     yxxrrgbb_idx = coord_transform(y_index, xx_index, rr_index, g_index, bb_index) 
#     yyxrrgbb_idx = coord_transform(yy_index, x_index, rr_index, g_index, bb_index) 
#     yyxxrrgbb_idx = coord_transform(yy_index, xx_index, rr_index, g_index, bb_index) 
#     yxrggbb_idx = coord_transform(y_index, x_index, r_index, gg_index, bb_index) 
#     yxxrggbb_idx = coord_transform(y_index, xx_index, r_index, gg_index, bb_index)
#     yyxrggbb_idx = coord_transform(yy_index, x_index, r_index, gg_index, bb_index) 
#     yyxxrggbb_idx = coord_transform(yy_index, xx_index, r_index, gg_index, bb_index)
#     yxrrggbb_idx = coord_transform(y_index, x_index, rr_index, gg_index, bb_index) 
#     yxxrrggbb_idx = coord_transform(y_index, xx_index, rr_index, gg_index, bb_index) 
#     yyxrrggbb_idx = coord_transform(yy_index, x_index, rr_index, gg_index, bb_index) 
#     yyxxrrggbb_idx = coord_transform(yy_index, xx_index, rr_index, gg_index, bb_index) 
    
#     # Shape (h x w x c, 2) because it is `data`, idx 0 is input, idx 1 is weight
#     array = array.reshape((-1, ))

#     return \
#     (1. - x_alpha) * (1. - y_alpha) * (1. - r_alpha) * (1. - g_alpha) * (1. - b_alpha) * array[yxrgb_idx]      + \
#     x_alpha        * (1. - y_alpha) * (1. - r_alpha) * (1. - g_alpha) * (1. - b_alpha) * array[yxxrgb_idx]     + \
#     (1. - x_alpha) * y_alpha        * (1. - r_alpha) * (1. - g_alpha) * (1. - b_alpha) * array[yyxrgb_idx]     + \
#     x_alpha        * y_alpha        * (1. - r_alpha) * (1. - g_alpha) * (1. - b_alpha) * array[yyxxrgb_idx]    + \
#     (1. - x_alpha) * (1. - y_alpha) * r_alpha        * (1. - g_alpha) * (1. - b_alpha) * array[yxrrgb_idx]     + \
#     x_alpha        * (1. - y_alpha) * r_alpha        * (1. - g_alpha) * (1. - b_alpha) * array[yxxrrgb_idx]    + \
#     (1. - x_alpha) * y_alpha        * r_alpha        * (1. - g_alpha) * (1. - b_alpha) * array[yyxrrgb_idx]    + \
#     x_alpha        * y_alpha        * r_alpha        * (1. - g_alpha) * (1. - b_alpha) * array[yyxxrrgb_idx]   + \
#     (1. - x_alpha) * (1. - y_alpha) * (1. - r_alpha) * g_alpha        * (1. - b_alpha) * array[yxrggb_idx]     + \
#     x_alpha        * (1. - y_alpha) * (1. - r_alpha) * g_alpha        * (1. - b_alpha) * array[yxxrggb_idx]    + \
#     (1. - x_alpha) * y_alpha        * (1. - r_alpha) * g_alpha        * (1. - b_alpha) * array[yyxrggb_idx]    + \
#     x_alpha        * y_alpha        * (1. - r_alpha) * g_alpha        * (1. - b_alpha) * array[yyxxrggb_idx]   + \
#     (1. - x_alpha) * (1. - y_alpha) * r_alpha        * g_alpha        * (1. - b_alpha) * array[yxrrggb_idx]    + \
#     x_alpha        * (1. - y_alpha) * r_alpha        * g_alpha        * (1. - b_alpha) * array[yxxrrggb_idx]   + \
#     (1. - x_alpha) * y_alpha        * r_alpha        * g_alpha        * (1. - b_alpha) * array[yyxrrggb_idx]   + \
#     x_alpha        * y_alpha        * r_alpha        * g_alpha        * (1. - b_alpha) * array[yyxxrrggb_idx]  + \
#     (1. - x_alpha) * (1. - y_alpha) * (1. - r_alpha) * (1. - g_alpha) * b_alpha        * array[yxrgbb_idx]     + \
#     x_alpha        * (1. - y_alpha) * (1. - r_alpha) * (1. - g_alpha) * b_alpha        * array[yxxrgbb_idx]    + \
#     (1. - x_alpha) * y_alpha        * (1. - r_alpha) * (1. - g_alpha) * b_alpha        * array[yyxrgbb_idx]    + \
#     x_alpha        * y_alpha        * (1. - r_alpha) * (1. - g_alpha) * b_alpha        * array[yyxxrgbb_idx]   + \
#     (1. - x_alpha) * (1. - y_alpha) * r_alpha        * (1. - g_alpha) * b_alpha        * array[yxrrgbb_idx]    + \
#     x_alpha        * (1. - y_alpha) * r_alpha        * (1. - g_alpha) * b_alpha        * array[yxxrrgbb_idx]   + \
#     (1. - x_alpha) * y_alpha        * r_alpha        * (1. - g_alpha) * b_alpha        * array[yyxrrgbb_idx]   + \
#     x_alpha        * y_alpha        * r_alpha        * (1. - g_alpha) * b_alpha        * array[yyxxrrgbb_idx]  + \
#     (1. - x_alpha) * (1. - y_alpha) * (1. - r_alpha) * g_alpha        * b_alpha        * array[yxrggbb_idx]    + \
#     x_alpha        * (1. - y_alpha) * (1. - r_alpha) * g_alpha        * b_alpha        * array[yxxrggbb_idx]   + \
#     (1. - x_alpha) * y_alpha        * (1. - r_alpha) * g_alpha        * b_alpha        * array[yyxrggbb_idx]   + \
#     x_alpha        * y_alpha        * (1. - r_alpha) * g_alpha        * b_alpha        * array[yyxxrggbb_idx]  + \
#     (1. - x_alpha) * (1. - y_alpha) * r_alpha        * g_alpha        * b_alpha        * array[yxrrggbb_idx]   + \
#     x_alpha        * (1. - y_alpha) * r_alpha        * g_alpha        * b_alpha        * array[yxxrrggbb_idx]  + \
#     (1. - x_alpha) * y_alpha        * r_alpha        * g_alpha        * b_alpha        * array[yyxrrggbb_idx]  + \
#     x_alpha        * y_alpha        * r_alpha        * g_alpha        * b_alpha        * array[yyxxrrggbb_idx]
