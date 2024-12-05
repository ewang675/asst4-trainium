import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A fused convolution - maxpool kernel that you need to implement for Part 2.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.
    pool_size: the size of the pool filter and pool stride.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: pool_size == 1 || pool_size == 2
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height // pool_size
out_pool_width = out_width // pool_size

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""

@nki.jit
def fused_conv2d_maxpool(X, W, bias, pool_size=1):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size

    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    MAX_FREE_DIM = 65536 # maximum FREE_DIM on sbuf tiles (physical constraint)

    # Various tiling dimensions (You may want to define more of them)
    TILE_C_OUT = min(out_channels, nl.tile_size.gemm_stationary_fmax) 
    n_tiles_c_out = out_channels // TILE_C_OUT
    TILE_C_IN = min(in_channels, nl.tile_size.pmax)
    n_tiles_c_in = in_channels // TILE_C_IN

    # TILE_HW is the number of output pixels per psum tile, given nl.tile_size.gemm_moving_fmax
    TILE_HW = (pool_size * out_width) * (nl.tile_size.gemm_moving_fmax // (pool_size * out_width)) 
    TILE_HW = min(TILE_HW, out_height * out_width)
    n_vert_pools = TILE_HW // (pool_size * out_width) # number of rows of output per psum tile
    TILE_H = TILE_HW // out_width # number of rows of input per psum tile

    # S_TILE_HW is the largest number of input elements that will fit into one s_tile in sbuf, given MAX_FREE_DIM
    n_conv_rows = MAX_FREE_DIM // (input_width * n_tiles_c_in) # number of input rows in one s_tile
    n_conv_rows = min(n_conv_rows, input_height)
    n_tiles_hw = (n_conv_rows - (filter_height - 1)) // TILE_H # number of input psum tiles per s_tile
    while out_height % (TILE_H * n_tiles_hw) != 0: # ensure s_tiles perfectly tile the image
        n_tiles_hw -= 1
    n_conv_rows = n_tiles_hw * TILE_H
    S_TILE_H = n_conv_rows + (filter_height - 1) # number of rows of input per s_tile
    S_TILE_HW = S_TILE_H * input_width * n_tiles_c_in
    n_tiles_s_in = out_height // (TILE_H * n_tiles_hw) # number of s_tiles per image

    # process the images in batches
    for b in nl.affine_range(batch_size):

        # bias fits entirely into sbuf
        bias_tiles = nl.ndarray((TILE_C_OUT, n_tiles_c_out), dtype=bias.dtype, buffer=nl.sbuf)
        for n in nl.affine_range(n_tiles_c_out):
            bias_tiles[:, n] = nl.load(bias[n * TILE_C_OUT:(n + 1) * TILE_C_OUT])
        # W fits entirely into sbuf
        weight_tiles = nl.ndarray((n_tiles_c_in, nl.par_dim(TILE_C_IN), out_channels, filter_height, filter_width), dtype=bias.dtype, buffer=nl.sbuf)
        for k in nl.affine_range(n_tiles_c_in):
            for l in nl.affine_range(out_channels):
                weight_tiles[k, :, l, :, :] = nl.load(W[l, k * TILE_C_IN:(k + 1) * TILE_C_IN, :, :])
        
        # process the image over s_tiles
        for s in nl.affine_range(n_tiles_s_in):
            # load in input
            s_tile = nl.ndarray((n_tiles_c_in, nl.par_dim(TILE_C_IN), S_TILE_H, input_width), dtype=X.dtype, buffer=nl.sbuf)
            for k in nl.affine_range(n_tiles_c_in):
                s_tile[k, :, :, :] = nl.load(X[b, k * TILE_C_IN:(k + 1) * TILE_C_IN, s * n_conv_rows:(s * n_conv_rows + S_TILE_H), :])
            
            # process the s_tile over batches of 128 output channels
            for n in nl.affine_range(n_tiles_c_out):
                # copy in bias tile from SBUF
                bias_ = nl.ndarray((TILE_C_OUT,1,1), dtype=bias.dtype, buffer=nl.sbuf)
                bias_[...] = nl.copy(bias_tiles[:, n])
                broadcasted_bias = bias_.broadcast_to((TILE_C_OUT, n_vert_pools, out_pool_width))
                # copy in weight tile from SBUF
                weight_ = nl.ndarray((n_tiles_c_in, nl.par_dim(TILE_C_IN), TILE_C_OUT, filter_height, filter_width), dtype=W.dtype, buffer=nl.sbuf)
                for k in nl.affine_range(n_tiles_c_in):
                    weight_[k, :, :, :, :] = nl.copy(weight_tiles[k, :, n * TILE_C_OUT:(n + 1) * TILE_C_IN, :, :])
                
                # process the s_tile over strips of input (< 512 pixels)that will fit into psum
                for m in nl.affine_range(n_tiles_hw):
                    conv_result = nl.zeros((TILE_C_OUT, TILE_H * out_width), dtype=X_out.dtype, buffer=nl.sbuf)
                    # copy in input tile strip from s_tile
                    input_ = nl.ndarray((n_tiles_c_in, nl.par_dim(TILE_C_IN), TILE_H + filter_height - 1, out_width + filter_width - 1), dtype=X.dtype, buffer=nl.sbuf)
                    for k in nl.affine_range(n_tiles_c_in):
                        for h in nl.affine_range(TILE_H + filter_height - 1):
                            true_h = m * TILE_H + h
                            input_[k, :, h, :] = nl.copy(s_tile[k, :, true_h, :])
                    
                    # compute convolution over this psum tile
                    for i in nl.affine_range(filter_height):
                        for j in nl.affine_range(filter_width):
                            res_psum = nl.zeros((TILE_C_OUT, TILE_H * out_width), nl.float32, buffer=nl.psum)
                            # process the tile over batches of 128 input channels
                            for k in nl.affine_range(n_tiles_c_in):
                                # shift the input tile by (i, j)
                                rhs_tile = nl.ndarray((TILE_C_IN, TILE_H * out_width), dtype=X.dtype, buffer=nl.sbuf)
                                for h in nl.affine_range(TILE_H):
                                    rhs_tile[:, h*out_width:(h+1)*out_width] = nl.copy(input_[k, :, h + i, j:j+out_width], dtype=X.dtype)
                                # multiply by the weight at filter position (i, j)
                                res_psum += nl.matmul(weight_[k, :, :, i, j], rhs_tile, transpose_x=True)
                            conv_result += nl.copy(res_psum, dtype=X_out.dtype)
                    
                    # reshape convolution output to X_out dimensions
                    conv_result_reshaped = nl.ndarray((TILE_C_OUT, TILE_H, out_width), dtype=X.dtype, buffer=nl.sbuf)
                    for h in nl.affine_range(TILE_H):
                        conv_result_reshaped[:, h, :] = conv_result[:, h * out_width:(h + 1) * out_width]

                    if pool_size > 1:
                        # with maxpool
                        i_0 = nl.arange(TILE_C_OUT)[:, None, None, None, None]
                        i_1 = nl.arange(n_vert_pools)[None, :, None, None, None]
                        i_2 = nl.arange(pool_size)[None, None, :, None, None]
                        i_3 = nl.arange(out_pool_width)[None, None, None, :, None]
                        i_4 = nl.arange(pool_size)[None, None, None, None, :]
                        out_tile = nl.max(conv_result_reshaped[i_0, pool_size * i_1 + i_2, pool_size * i_3 + i_4], axis=[2, 4])
                        out_tile = out_tile + broadcasted_bias
                        tile_idx = n_tiles_hw * s + m
                        nl.store(X_out[b, n * TILE_C_OUT:(n + 1) * TILE_C_OUT, tile_idx * n_vert_pools:(tile_idx + 1) * n_vert_pools, :], value=out_tile)
                    else:
                        # without maxpool
                        out_tile = conv_result_reshaped + broadcasted_bias
                        tile_idx = n_tiles_hw * s + m
                        nl.store(X_out[b, n * TILE_C_OUT:(n + 1) * TILE_C_OUT, tile_idx * TILE_H:(tile_idx + 1) * TILE_H, :], value=out_tile)

    return X_out
