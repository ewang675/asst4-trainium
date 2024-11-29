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

    # Various tiling dimensions (You may want to define more of them)
    hw_dim = input_height * input_width
    TILE_HW = min(hw_dim, nl.tile_size.gemm_stationary_fmax)
    n_tiles_hw = hw_dim // TILE_HW

    TILE_C_IN = min(in_channels, nl.tile_size.pmax)
    n_tiles_c_in = in_channels // TILE_C_IN

    TILE_C_OUT = min(out_channels, nl.tile_size.gemm_moving_fmax) 
    n_tiles_c_out = out_channels // TILE_C_OUT


    # Process the images in batches
    for b in nl.affine_range(batch_size):
        # TODO: Perform the convolution of X[b] with the weights W and bias b, followed by a maxpool
        # and store the result in X_out[b]
        
        # convolution first 
        conv_result = nl.zeros((out_height, out_width, out_channels), nl.float32, buffer=nl.psum)
        for i in nl.affine_range(filter_height):
            for j in nl.affine_range(filter_width):

                # matrix multiplication
                matmul_result = nl.zeros((input_height, input_width, out_channels), nl.float32, buffer=nl.psum)
                for m in nl.affine_range(n_tiles_hw):
                    for n in nl.affine_range(n_tiles_c_out):
                        # Allocate a tensor in PSUM
                        res_psum = nl.zeros((TILE_HW, TILE_C_OUT), nl.float32, buffer=nl.psum)

                        for k in nl.affine_range(n_tiles_c_in):
                            # Declare the tiles on SBUF
                            lhsT_tile = nl.ndarray((TILE_C_IN, TILE_HW), dtype=X.dtype, buffer=nl.sbuf)
                            rhs_tile = nl.ndarray((TILE_C_IN, TILE_C_OUT), dtype=W.dtype, buffer=nl.sbuf)

                            # Load tiles from lhsT and rhs
                            # Using a slow load rn, see if there is a reshape possibility
                            for hw in nl.affine_range(TILE_HW):
                                true_hw = hw + m * TILE_HW
                                h = true_hw // input_width
                                w = true_hw % input_width
                                lhsT_tile[:, hw] = nl.load(X[b, k * TILE_C_IN:(k + 1) * TILE_C_IN, h + i, w + j])
                            rhs_tile[...] = nl.load(W[n * TILE_C_OUT:(n + 1) * TILE_C_OUT, k * TILE_C_IN:(k + 1) * TILE_C_IN, i, j])

                            # Accumulate partial-sums into PSUM
                            res_psum += nl.matmul(lhsT_tile[...], rhs_tile[...], transpose_x=True)

                        for hw in nl.affine_range(TILE_HW):
                            true_hw = hw + m * TILE_HW
                            h = true_hw // input_width
                            w = true_hw % input_width
                            matmul_result[h, w, n * TILE_C_OUT:(n + 1) * TILE_C_OUT] = res_psum[hw, :]

                conv_result[i:out_height, j:out_width, :] += matmul_result[0:out_height - i, 0:out_width - j, :]
                        # Copy the result from PSUM back to SBUF, and cast to expected output data-type
                        # res_sb = nl.copy(res_psum, dtype=result.dtype)
                        # nl.store(result[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N],
                        #         value=res_sb)
        
        # maxpooling
        for i in nl.affine_range(out_pool_height):
            for j in nl.affine_range(out_pool_width):
                for c in nl.affine_range(out_channels):
                    max_val = nl.float32(-np.inf)
                    for m in nl.affine_range(pool_size):
                        for n in nl.affine_range(pool_size):
                            max_val = nl.max(max_val, conv_result[i * pool_size + m, j * pool_size + n, c])
                    X_out[b, c, i, j] = max_val

    return X_out