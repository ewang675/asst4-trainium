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
    # assert in_channels % 128 == 0

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
    print(f"TILE_C_OUT: {TILE_C_OUT}, n_tiles_c_out: {n_tiles_c_out}, TILE_C_IN: {TILE_C_IN}, n_tiles_c_in: {n_tiles_c_in}")

    # TILE_HW is the largest multiple of pool_size * input_width that is less than or equal to nl.tile_size.gemm_moving_fmax
    # I assume that pool_size * input_width is always less than or equal to nl.tile_size.gemm_moving_fmax
    print(f"pool_size: {pool_size}, input_width: {input_width}, gemm_moving: {nl.tile_size.gemm_moving_fmax}")
    TILE_HW = (pool_size * out_width) * (nl.tile_size.gemm_moving_fmax // (pool_size * out_width))
    TILE_HW = min(TILE_HW, out_height * out_width)
    n_tiles_hw = out_height * out_width // TILE_HW
    n_vert_pools = TILE_HW // (pool_size * out_width)
    TILE_H = TILE_HW // out_width
    print(f"TILE_HW: {TILE_HW}, n_tiles_hw: {n_tiles_hw}, n_vert_pools: {n_vert_pools}, TILE_H: {TILE_H}")

    # nl.device_print("X", X)

    # Process the images in batches
    for b in nl.affine_range(batch_size):
        # TODO: Perform the convolution of X[b] with the weights W and bias b, followed by a maxpool
        # and store the result in X_out[b]
        
        # convolution first 
        for m in nl.affine_range(n_tiles_hw):
            for n in nl.affine_range(n_tiles_c_out):
                conv_result = nl.zeros((TILE_C_OUT, TILE_H * out_width), nl.float32, buffer=nl.psum)
                if m == n_tiles_hw - 1: # special case for the last tile
                    # we need to ignore the last few rows since they are out of bounds
                    for h in nl.affine_range(TILE_H - filter_height + 1):
                        true_h = m * TILE_H + h
                        for i in nl.affine_range(filter_height):
                            for j in nl.affine_range(filter_width):
                                # Matrix multiplication
                                for k in nl.affine_range(n_tiles_c_in):
                                    # Declare the tiles on SBUF
                                    lhsT_tile = nl.ndarray((TILE_C_IN, TILE_C_OUT), dtype=W.dtype, buffer=nl.sbuf)
                                    rhs_tile = nl.ndarray((TILE_C_IN, out_width), dtype=X.dtype, buffer=nl.sbuf)

                                    for l in nl.affine_range(TILE_C_OUT):
                                        lhsT_tile[:, l] = nl.load(W[n + l, k * TILE_C_IN:(k + 1) * TILE_C_IN, i, j])
                                    rhs_tile[...] = nl.load(X[b, k * TILE_C_IN:(k + 1) * TILE_C_IN, true_h + i, j:j+out_width])

                                    # nl.device_print("lhsT_tile", lhsT_tile)
                                    # nl.device_print("rhs_tile", rhs_tile)

                                    # Accumulate partial-sums into PSUM
                                    conv_result[:,h*out_width:(h+1)*out_width] += nl.matmul(lhsT_tile[...], rhs_tile[...], transpose_x=True)
                                    nl.device_print("h", h)
                                    # nl.device_print("conv_result", conv_result)
                else:
                    for h in nl.affine_range(TILE_H):
                        true_h = m * TILE_H + h
                        for i in nl.affine_range(filter_height):
                            for j in nl.affine_range(filter_width):
                                # Matrix multiplication
                                for k in nl.affine_range(n_tiles_c_in):
                                    # Declare the tiles on SBUF
                                    lhsT_tile = nl.ndarray((TILE_C_IN, TILE_C_OUT), dtype=W.dtype, buffer=nl.sbuf)
                                    rhs_tile = nl.ndarray((TILE_C_IN, out_width), dtype=X.dtype, buffer=nl.sbuf)

                                    lhsT_tile[...] = nl.load(W[i, j, n * TILE_C_OUT:(n + 1) * TILE_C_OUT, k * TILE_C_IN:(k + 1) * TILE_C_IN])
                                    rhs_tile[...] = nl.load(X[b, k * TILE_C_IN:(k + 1) * TILE_C_IN, true_h + i, j:j+out_width])

                                    # Accumulate partial-sums into PSUM
                                    conv_result[:,h*out_width:(h+1)*out_width] += nl.matmul(lhsT_tile[...], rhs_tile[...], transpose_x=True)

                # Note that we will need to ignore all hw in TILE_HW with w >= out_width and h >= out_height
                if pool_size > 1:
                    for h in nl.affine_range(n_vert_pools):
                        for w in nl.affine_range(out_pool_width):
                            true_h = m * n_vert_pools + h
                            maxpool_result = nl.zeros((TILE_C_OUT,1), nl.float32, buffer=nl.psum)
                            for i in nl.affine_range(pool_size):
                                for j in nl.affine_range(pool_size):
                                    hw_h = h * pool_size + i
                                    hw_w = w * pool_size + j
                                    assert (hw_h < out_height and hw_w < out_width)
                                    maxpool_result = nl.maximum(maxpool_result, conv_result[:, hw_h * out_width + hw_w])
                            # Copy the result from PSUM back to SBUF, and cast to expected output data-type
                            res_sb = nl.copy(maxpool_result, dtype=X_out.dtype)
                            nl.store(X_out[b, n * TILE_C_OUT:(n + 1) * TILE_C_OUT, true_h, w],
                                    value=res_sb)  
                else:
                    print("no maxpooling")
                    for h in nl.affine_range(n_vert_pools):
                        for w in nl.affine_range(out_pool_width):
                            true_h = m * n_vert_pools + h
                            # Copy the result from PSUM back to SBUF, and cast to expected output data-type
                            res_sb = nl.copy(conv_result[:, h * out_width + w], dtype=X_out.dtype)
                            nl.store(X_out[b, n * TILE_C_OUT:(n + 1) * TILE_C_OUT, true_h, w],
                                    value=res_sb)    

    return X_out