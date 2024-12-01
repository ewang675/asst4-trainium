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

    # wff_i_0 = nl.arange(filter_height)[:, None, None, None]
    # wff_i_1 = nl.arange(filter_width)[None, :, None, None]
    # wff_i_2 = nl.arange(in_channels)[None, None, :, None]
    # W_ff = nl.ndarray((filter_height, filter_width, in_channels, out_channels), dtype=W.dtype, buffer=nl.sbuf)
    # for o in nl.affine_range(out_channels):
    #     W_ff[wff_i_0, wff_i_1, wff_i_2, o] = nl.load(W[o, wff_i_2, wff_i_0, wff_i_1])
    # print("W shape: ", W.shape)
    # print("W_ff shape: ", W_ff.shape)
    # print(f"out_height: {out_height}, out_width: {out_width}, out_pool_height: {out_pool_height}, out_pool_width: {out_pool_width}")
    
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

    # Various tiling dimensions (You may want to define more of them)
    TILE_C_OUT = min(out_channels, nl.tile_size.gemm_stationary_fmax) 
    n_tiles_c_out = out_channels // TILE_C_OUT

    TILE_C_IN = min(in_channels, nl.tile_size.pmax)
    n_tiles_c_in = in_channels // TILE_C_IN
    # print(f"TILE_C_OUT: {TILE_C_OUT}, n_tiles_c_out: {n_tiles_c_out}, TILE_C_IN: {TILE_C_IN}, n_tiles_c_in: {n_tiles_c_in}")

    # TILE_HW is the largest multiple of pool_size * input_width that is less than or equal to nl.tile_size.gemm_moving_fmax
    # I assume that pool_size * input_width is always less than or equal to nl.tile_size.gemm_moving_fmax
    # print(f"pool_size: {pool_size}, input_width: {input_width}, gemm_moving: {nl.tile_size.gemm_moving_fmax}")
    TILE_HW = (pool_size * out_width) * (nl.tile_size.gemm_moving_fmax // (pool_size * out_width))
    TILE_HW = min(TILE_HW, out_height * out_width)
    n_tiles_hw = out_height * out_width // TILE_HW
    n_vert_pools = TILE_HW // (pool_size * out_width)
    TILE_H = TILE_HW // out_width
    # print(f"TILE_HW: {TILE_HW}, n_tiles_hw: {n_tiles_hw}, n_vert_pools: {n_vert_pools}, TILE_H: {TILE_H}")

    # Process the images in batches
    for b in nl.affine_range(batch_size):
        # TODO: Perform the convolution of X[b] with the weights W and bias b, followed by a maxpool
        # and store the result in X_out[b]
        
        # convolution first 
        for n in nl.affine_range(n_tiles_c_out):
            bias_tile = nl.ndarray((TILE_C_OUT,1,1), dtype=bias.dtype, buffer=nl.sbuf)
            bias_tile[...] = nl.load(bias[n * TILE_C_OUT:(n + 1) * TILE_C_OUT])
            broadcasted_bias = bias_tile.broadcast_to((TILE_C_OUT, n_vert_pools, out_pool_width))
            for m in nl.affine_range(n_tiles_hw):
                conv_result = nl.zeros((TILE_C_OUT, TILE_H * out_width), dtype=X_out.dtype, buffer=nl.sbuf)
                for i in nl.affine_range(filter_height):
                    for j in nl.affine_range(filter_width):
                        # Matrix multiplication    
                        res_psum = nl.zeros((TILE_C_OUT, TILE_H * out_width), nl.float32, buffer=nl.psum)
                        for k in nl.affine_range(n_tiles_c_in):
                            # Declare the tiles on SBUF
                            lhsT_tile = nl.ndarray((TILE_C_IN, TILE_C_OUT), dtype=W.dtype, buffer=nl.sbuf)
                            # i_0 = nl.arange(TILE_C_IN)[:, None]
                            # i_1 = nl.arange(TILE_C_OUT)[None, :]
                            # lhsT_tile[...] = W_ff[i, j, k * TILE_C_IN:(k + 1) * TILE_C_IN, n * TILE_C_OUT:(n + 1) * TILE_C_OUT]
                            for l in nl.affine_range(TILE_C_OUT):
                                lhsT_tile[:, l] = nl.load(W[n * TILE_C_OUT + l, k * TILE_C_IN:(k + 1) * TILE_C_IN, i, j])

                            rhs_tile = nl.ndarray((TILE_C_IN, TILE_H * out_width), dtype=X.dtype, buffer=nl.sbuf)
                            # i_1 = nl.arange(TILE_C_IN)[None, :, None, None]
                            # i_2 = nl.arange(TILE_H)[None, None, :, None]
                            # i_3 = nl.arange(out_width)[None, None, None, :]
                            # rhs_tile[i_1, i_2 * out_width + i_3] = nl.load(X[b, k * TILE_C_IN + i_1, i + m * TILE_H + i_2, j + i_3])
                            for h in nl.affine_range(TILE_H):
                                true_h = m * TILE_H + h
                                rhs_tile[:, h*out_width:(h+1)*out_width] = nl.load(X[b, k * TILE_C_IN:(k + 1) * TILE_C_IN, true_h + i, j:j+out_width])
                                
                            # Accumulate partial-sums into PSUM
                            res_psum += nl.matmul(lhsT_tile[...], rhs_tile[...], transpose_x=True)
                        # Copy the result from PSUM back to SBUF, and cast to expected output data-type
                        conv_result += nl.copy(res_psum, dtype=X_out.dtype)

                if pool_size > 1:
                    i_0 = nl.arange(TILE_C_OUT)[:, None, None, None, None]
                    i_1 = nl.arange(n_vert_pools)[None, :, None, None, None]
                    i_2 = nl.arange(pool_size)[None, None, :, None, None]
                    i_3 = nl.arange(out_pool_width)[None, None, None, :, None]
                    i_4 = nl.arange(pool_size)[None, None, None, None, :]
                    out_tile = nl.max(conv_result[i_0, (i_1 * pool_size + i_2) * out_width + i_3 * pool_size + i_4], axis=[2, 4])
                    out_tile += broadcasted_bias
                    nl.store(X_out[b, n * TILE_C_OUT:(n + 1) * TILE_C_OUT, m * n_vert_pools:(m + 1) * n_vert_pools, :], value=out_tile)
                else:
                    # print("no maxpooling, assume out_pool_width = out_width")
                    i_0 = nl.arange(TILE_C_OUT)[:, None, None]
                    i_1 = nl.arange(n_vert_pools)[None, :, None]
                    i_2 = nl.arange(out_pool_width)[None, None, :]
                    out_tile = conv_result[i_0, i_1 * out_pool_width + i_2]
                    out_tile += broadcasted_bias
                    nl.store(X_out[b, n * TILE_C_OUT:(n + 1) * TILE_C_OUT, m * n_vert_pools:(m + 1) * n_vert_pools, :], value=out_tile)

    return X_out