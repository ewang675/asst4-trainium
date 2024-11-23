import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal

def shift(input_tensor, shift_amount):
    """
    Shifts the input tensor spatially by the given shift amount.

    Args:
        input_tensor (ndarray): The input tensor of shape (height, width, channels).
        shift_amount (tuple): Tuple (i, j) indicating the number of rows and columns to shift.
    
    Returns:
        ndarray: The shifted tensor, padded with zeros.
    """
    i, j = shift_amount
    h, w, c = input_tensor.shape
    
    # Create an output tensor filled with zeros
    shifted_tensor = np.zeros_like(input_tensor)
    
    # Compute valid slicing ranges
    start_row, end_row = max(0, i), min(h, h + i)
    start_col, end_col = max(0, j), min(w, w + j)
    
    src_start_row, src_end_row = max(0, -i), min(h, h - i)
    src_start_col, src_end_col = max(0, -j), min(w, w - j)
    
    # Copy valid region into the shifted tensor
    shifted_tensor[start_row:end_row, start_col:end_col, :] = input_tensor[src_start_row:src_end_row, src_start_col:src_end_col, :]
    
    return shifted_tensor

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
    c_in_pmax = nl.tile_size.pmax
    n_tiles_c_in = in_channels // c_in_pmax

    # Process the images in batches
    for b in nl.affine_range(batch_size):
        # TODO: Perform the convolution of X[b] with the weights W and bias b, followed by a maxpool
        # and store the result in X_out[b]

        FREE_DIM = 1000
        PARTITION_DIM = 128
        TILE_M = PARTITION_DIM * FREE_DIM * in_channels
        M = input_height * input_width

        # Allocate space for X in HBM.
        X_b = nl.ndarray((M // TILE_M, in_channels, PARTITION_DIM, FREE_DIM), dtype=X.dtype, buffer=nl.hbm)

        # Load X[b] in.
        X_b = nl.load(X[b])

        for m in nl.affine_range((M // TILE_M)):
            # Load in this tile
            X_b_tile = nl.ndarray((in_channels, PARTITION_DIM, FREE_DIM), dtype=X_b.dtype, buffer=nl.sbuf)
            X_b_tile = nl.load(X_b[m])
            
            res = np.zeros(out_channels, PARTITION_DIM, FREE_DIM)

            for i in range(filter_height):
                for j in range(filter_width):
                    input_shifted = shift(X_b_tile, (i, j))
                    res += matmul(input_shifted, W[i, j, :, :])
        
        res += bias

        nl.store(X_out[b], value=out_b)

    return X_out

