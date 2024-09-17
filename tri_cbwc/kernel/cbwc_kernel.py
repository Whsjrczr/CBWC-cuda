import torch
import triton
import triton.language as tl

if hasattr(tl, "libdevice"):
    tl_math = tl.libdevice
else:
    tl_math = tl.math

@triton.jit
def _cbwc_fwd_fused(
    X,
    Y,
    W,
    B,
    stride,
    N,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel invocation for forward pass of RMS normalization with fused operations

    Params:
        - X (tensor): Input tensor
        - Y (tensor): Output tensor where the results will be written
        - W (tensor): Column based weight centered weight tensor, which will used in calculation
        - B (tensor): Bias tensor added to the scaled input
        - stride (int): Stride to be applied when accessing elements in the input and output tensors
        - N (int): Number of input feature
        - M (int): Number of output feature
        - BLOCK_SIZE (constexpr): Size of the block for computation, provided as a compile-time constant

    Return:
        - None

    Usage:
        _cbwc_fwd_fused[grid, block](X, Y, W, B, stride, N, BLOCK_SIZE)
    """
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride

    _rms = 0
    _rms = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _rms += a * a
    rms = tl.sqrt(tl.sum(_rms) / N + eps)

    tl.store(Rstd + row, rms)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
        x_hat = x / rms
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)


@triton.jit
def _rms_norm_bwd_dx_fused(
    DX,
    DY,
    DV,
    DW,
    DB,
    X,
    W,
    B,
    Rstd,
    Lock,
    stride,
    N,
    eps,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Kernel invocation for backward pass of RMS normalization, computing gradients w.r.t. input

    Params:
        - DX (tensor): Gradient of the loss with respect to the inputs
        - DY (tensor): Gradient of the loss with respect to the outputs
        - DV (tensor): Gradient of the loss with respect to the origin scale tensor V
        - DW (tensor): Gradient of the loss with respect to the scale tensor W
        - DB (tensor): Gradient of the loss with respect to the bias tensor B
        - X (tensor): Input tensor from the forward pass
        - W (tensor): Scale tensor applied during the forward pass
        - B (tensor): Bias tensor added during the forward pass
        - Lock (tensor): Lock tensor for atomic operations to prevent race conditions
        - stride (int): Stride to be applied when accessing elements in the tensors
        - N (int): Number of elements in each tensor
        - GROUP_SIZE_M (constexpr): Size of the group for M dimension, provided as a compile-time constant
        - BLOCK_SIZE_N (constexpr): Size of the block for N dimension, provided as a compile-time constant

    Return:
        - None

    Usage:
        _rms_norm_bwd_dx_fused[grid, block](DX, DY, DW, DB, X, W, B, Rstd, Lock, stride, N, eps, GROUP_SIZE_M, BLOCK_SIZE_N)
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW = DW + lock_id * N + cols
    DB = DB + lock_id * N + cols
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    rstd = tl.load(Rstd + row)
    x_norm = x * rstd
    wdy = w * dy
    dx = wdy * rstd
    tl.store(DX + cols, dx, mask=mask)
    partial_dw = (dy * x_norm).to(w.dtype)
    partial_db = dy.to(w.dtype)

    # Locking mechanism to prevent race conditions
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
        partial_db += tl.load(DB, mask=mask)

    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)
    tl.atomic_xchg(Lock, 0)
