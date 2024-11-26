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

    out_size = out_pool_height * out_pool_width
    filter_size = filter_height * filter_width

    # Various tiling dimensions (You may want to define more of them)
    c_in_pmax = nl.tile_size.pmax
    n_tiles_c_in = in_channels // c_in_pmax

    # Process the images in batches
    for b in nl.affine_range(batch_size):
        # TODO: Perform the convolution of X[b] with the weights W and bias b, followed by a maxpool
        # and store the result in X_out[b]

        # ASSUMING FOR NOW that everything fits in SBUF.

        # for i in range((input_height * input_width) // (filter_height * filter_width)):
        
        # Construct A.
        A = nl.ndarray((filter_size * in_channels, out_size), dtype=X.dtype, buffer=nl.hbm)
        print(A.shape)
        for i in range(out_pool_height):
            for j in range(out_pool_width):
                p = i * out_pool_width + j
                temp_buf = nl.ndarray((in_channels, filter_height, filter_width), dtype=X.dtype, buffer=nl.sbuf)
                temp_buf[...] = nl.load(X[b, :, i:(i + filter_height), j:(j + filter_width)])
                nl.store(A[:, p], value=temp_buf.reshape((filter_size * in_channels,)))

        # Construct B.
        B = nl.ndarray((filter_size * in_channels, out_channels), dtype=X.dtype, buffer=nl.hbm)
        temp_buf = nl.ndarray((filter_size * in_channels, out_channels), dtype=X.dtype, buffer=nl.sbuf)
        temp_buf[...] = nl.load(W.reshape((filter_size * in_channels, out_channels)))
        nl.store(B, value=temp_buf)
        print(B.shape)
        
        # MATMUL, copied from assignment spec
        # Construct C. 
        C = nl.ndarray((out_size, out_channels), dtype=X_out.dtype, buffer=nl.hbm)
        print(C.shape)

        K, M = A.shape
        K_, N = B.shape
        assert K == K_, "lhsT and rhs must have the same contraction dimension"

        # Maximum free dimension of the stationary operand of general matrix multiplication on tensor engine
        TILE_M = min(M, nl.tile_size.gemm_stationary_fmax)  # 128

        # Maximum partition dimension of a tile
        TILE_K = min(K, nl.tile_size.pmax)  # 128

        # Maximum free dimension of the moving operand of general matrix multiplication on tensor engine
        TILE_N = min(N, nl.tile_size.gemm_moving_fmax)  # 512

        # Use affine_range to loop over tiles
        for m in nl.affine_range(M // TILE_M):
            for n in nl.affine_range(N // TILE_N):
                print("hello!\n")
                # Allocate a tensor in PSUM
                res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

                for k in nl.affine_range(K // TILE_K):
                    # Declare the tiles on SBUF
                    A_tile = nl.ndarray((TILE_K, TILE_M), dtype=A.dtype, buffer=nl.sbuf)
                    B_tile = nl.ndarray((TILE_K, TILE_N), dtype=B.dtype, buffer=nl.sbuf)

                    # Load tiles from lhsT and rhs
                    A_tile[...] = nl.load(A[k * TILE_K:(k + 1) * TILE_K,
                                                m * TILE_M:(m + 1) * TILE_M])
                    B_tile[...] = nl.load(B[k * TILE_K:(k + 1) * TILE_K,
                                                n * TILE_N:(n + 1) * TILE_N])

                    # Accumulate partial-sums into PSUM
                    res_psum += nl.matmul(A_tile[...], B_tile[...], transpose_x=True)

                # Copy the result from PSUM back to SBUF, and cast to expected output data-type
                res_sb = nl.copy(res_psum, dtype=C.dtype)
                nl.store(C[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N],
                        value=res_sb)
        
        temp_buf_ = nl.ndarray((out_channels, out_pool_height, out_pool_width), dtype=X_out.dtype, buffer=nl.sbuf)
        temp_buf_[...] = nl.load(C.reshape((out_channels, out_pool_height, out_pool_width)))
        nl.store(X_out[b], value=temp_buf_)

    return X_out