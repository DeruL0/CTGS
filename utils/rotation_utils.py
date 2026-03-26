import torch
import torch.nn.functional as F


def _as_batch_axis_angle(axis_angle: torch.Tensor):
    if axis_angle.shape[-1] != 3:
        raise ValueError(f"Expected axis-angle vectors with last dim 3, got {axis_angle.shape}.")
    squeeze = axis_angle.ndim == 1
    if squeeze:
        axis_angle = axis_angle.unsqueeze(0)
    return axis_angle, squeeze


def _as_batch_matrix(matrix: torch.Tensor):
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrices with trailing shape (3, 3), got {matrix.shape}.")
    squeeze = matrix.ndim == 2
    if squeeze:
        matrix = matrix.unsqueeze(0)
    return matrix, squeeze


def quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    if quaternion.shape[-1] != 4:
        raise ValueError(f"Expected quaternions with last dim 4, got {quaternion.shape}.")

    quaternion = F.normalize(quaternion, dim=-1)
    qw, qx, qy, qz = quaternion.unbind(dim=-1)

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    matrix = torch.stack(
        (
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            1.0 - 2.0 * (xx + yy),
        ),
        dim=-1,
    )
    return matrix.reshape(quaternion.shape[:-1] + (3, 3))


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    axis_angle, squeeze = _as_batch_axis_angle(axis_angle)

    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    half_angle = 0.5 * angle
    scale = 0.5 * torch.sinc(half_angle / torch.pi)
    quaternion = torch.cat((torch.cos(half_angle), axis_angle * scale), dim=-1)
    matrix = quaternion_to_matrix(quaternion)
    return matrix[0] if squeeze else matrix


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    matrix, squeeze = _as_batch_matrix(matrix)

    m00 = matrix[..., 0, 0]
    m01 = matrix[..., 0, 1]
    m02 = matrix[..., 0, 2]
    m10 = matrix[..., 1, 0]
    m11 = matrix[..., 1, 1]
    m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]
    m21 = matrix[..., 2, 1]
    m22 = matrix[..., 2, 2]

    qw = torch.empty_like(m00)
    qx = torch.empty_like(m00)
    qy = torch.empty_like(m00)
    qz = torch.empty_like(m00)

    trace = m00 + m11 + m22
    mask_trace = trace > 0.0

    if mask_trace.any():
        s = 2.0 * torch.sqrt(torch.clamp(trace[mask_trace] + 1.0, min=1e-12))
        qw[mask_trace] = 0.25 * s
        qx[mask_trace] = (m21[mask_trace] - m12[mask_trace]) / s
        qy[mask_trace] = (m02[mask_trace] - m20[mask_trace]) / s
        qz[mask_trace] = (m10[mask_trace] - m01[mask_trace]) / s

    mask_x = (~mask_trace) & (m00 > m11) & (m00 > m22)
    if mask_x.any():
        s = 2.0 * torch.sqrt(torch.clamp(1.0 + m00[mask_x] - m11[mask_x] - m22[mask_x], min=1e-12))
        qw[mask_x] = (m21[mask_x] - m12[mask_x]) / s
        qx[mask_x] = 0.25 * s
        qy[mask_x] = (m01[mask_x] + m10[mask_x]) / s
        qz[mask_x] = (m02[mask_x] + m20[mask_x]) / s

    mask_y = (~mask_trace) & (~mask_x) & (m11 > m22)
    if mask_y.any():
        s = 2.0 * torch.sqrt(torch.clamp(1.0 + m11[mask_y] - m00[mask_y] - m22[mask_y], min=1e-12))
        qw[mask_y] = (m02[mask_y] - m20[mask_y]) / s
        qx[mask_y] = (m01[mask_y] + m10[mask_y]) / s
        qy[mask_y] = 0.25 * s
        qz[mask_y] = (m12[mask_y] + m21[mask_y]) / s

    mask_z = (~mask_trace) & (~mask_x) & (~mask_y)
    if mask_z.any():
        s = 2.0 * torch.sqrt(torch.clamp(1.0 + m22[mask_z] - m00[mask_z] - m11[mask_z], min=1e-12))
        qw[mask_z] = (m10[mask_z] - m01[mask_z]) / s
        qx[mask_z] = (m02[mask_z] + m20[mask_z]) / s
        qy[mask_z] = (m12[mask_z] + m21[mask_z]) / s
        qz[mask_z] = 0.25 * s

    quaternion = torch.stack((qw, qx, qy, qz), dim=-1)
    quaternion = F.normalize(quaternion, dim=-1)
    quaternion = torch.where(quaternion[..., :1] < 0.0, -quaternion, quaternion)
    return quaternion[0] if squeeze else quaternion


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    quaternion = matrix_to_quaternion(matrix)
    squeeze = quaternion.ndim == 1
    if squeeze:
        quaternion = quaternion.unsqueeze(0)

    xyz = quaternion[..., 1:]
    sin_half_angle = torch.linalg.norm(xyz, dim=-1, keepdim=True)
    half_angle = torch.atan2(sin_half_angle, quaternion[..., :1])
    angle = 2.0 * half_angle

    scale = torch.where(
        sin_half_angle > 1e-8,
        angle / sin_half_angle,
        torch.full_like(sin_half_angle, 2.0),
    )
    axis_angle = xyz * scale
    return axis_angle[0] if squeeze else axis_angle
