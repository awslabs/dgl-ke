import torch as th

# some constants
MIN_NORM = 1e-15
BALL_EPS = {th.float32: 4e-3, th.float64: 1e-5}


# math in hyperbolic space

class Artanh(th.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        dtype = x.dtype
        x = x.double()
        return (th.log_(1 + x).sub_(th.log_(1 - x))).mul_(0.5).to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


def artanh(x):
    return Artanh.apply(x)


def tanh(x):
    return x.clamp(-15, 15).tanh()


def hyp_distance_multi_c(x, y, c, mode='batch'):
    sqrt_c = c ** 0.5
    if mode == 'mm':
        ynorm = th.norm(y, p=2, dim=-1, keepdim=True).transpose(1, 2)
        xy = th.bmm(x, y.transpose(1, 2)) / ynorm
        gamma = tanh(th.bmm(sqrt_c, ynorm)) / sqrt_c
    elif mode == 'batch':
        ynorm = th.norm(y, p=2, dim=-1, keepdim=True)
        xy = th.sum(x * y / ynorm, dim=-1, keepdim=True)
        gamma = tanh(sqrt_c * ynorm) / sqrt_c
    else:
        raise ValueError(f'{mode} mode is not supported. Choose from [batch, mm]')
    x2 = th.sum(x * x, dim=-1, keepdim=True)
    c1 = 1 - 2 * c * gamma * xy + c * gamma ** 2
    c2 = 1 - c * x2
    num = th.sqrt((c1 ** 2) * x2 + (c2 ** 2) * (gamma ** 2) - (2 * c1 * c2) * gamma * xy)
    denom = 1 - 2 * c * gamma * xy + (c ** 2) * (gamma ** 2) * x2
    pairwise_norm = num / denom.clamp_min(MIN_NORM)
    dist = artanh(sqrt_c * pairwise_norm)
    return 2 * dist / sqrt_c


def mobius_add(x, y, c):
    x2 = th.sum(x * x, dim=-1, keepdim=True)
    y2 = th.sum(y * y, dim=-1, keepdim=True)
    xy = th.sum(x * y, dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / denom.clamp_min(MIN_NORM)


def project(x, c):
    norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    eps = BALL_EPS[x.dtype]
    maxnorm = (1 - eps) / (c ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return th.where(cond, projected, x)


def expmap0(u, c):
    sqrt_c = c ** 0.5
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return project(gamma_1, c)


# euclidean space

def givens_rotations(r, x, comp='batch'):
    if comp == 'batch':
        givens = r.view((r.shape[0], -1, 2))
        givens = givens / th.norm(givens, p=2, dim=-1, keepdim=True)
        x = x.view((r.shape[0], -1, 2))
        x_rot = givens[:, :, 0:1] * x + givens[:, :, 1:] * th.cat((-x[:, :, 1:], x[:, :, 0:1]), dim=-1)
        return x_rot.view((r.shape[0], -1))
    else:
        givens = r.view((r.shape[0], r.shape[1], -1, 2))
        givens = givens / th.norm(givens, p=2, dim=-1, keepdim=True)
        x = x.view((x.shape[0], x.shape[1], -1, 2))
        x_rot_a = th.einsum('bcde,bnde->bcnde', givens[:, :, :, 0:1].expand(-1, -1, -1, 2), x)
        x_rot_b = th.einsum('bcde,bnde->bcnde', givens[:, :, :, 1:].expand(-1, -1, -1, 2),
                            th.cat((-x[:, :, :, 1:], x[:, :, :, 0:1]), dim=-1))
        x_rot = x_rot_a + x_rot_b
        return x_rot.view((r.shape[0], r.shape[1], x.shape[1], -1))


def givens_reflection(r, x, comp='batch'):
    if comp == 'batch':
        givens = r.view((r.shape[0], -1, 2))
        givens = givens / th.norm(givens, p=2, dim=-1, keepdim=True)
        x = x.view((r.shape[0], -1, 2))
        x_ref = givens[:, :, 0:1] * th.cat((x[:, :, 0:1], -x[:, :, 1:]), dim=-1) + givens[:, :, 1:] * th.cat(
            (x[:, :, 1:], x[:, :, 0:1]), dim=-1)
        return x_ref.view((r.shape[0], -1))
    else:
        givens = r.view((r.shape[0], r.shape[1], -1, 2))
        givens = givens / th.norm(givens, p=2, dim=-1, keepdim=True)
        x = x.view((x.shape[0], x.shape[1], -1, 2))
        x_ref_a = th.einsum('bcde,bnde->bcnde', givens[:, :, :, 0:1].expand(-1, -1, -1, 2),
                            th.cat((x[:, :, :, 0:1], -x[:, :, :, 1:]), dim=-1))
        x_ref_b = th.einsum('bcde,bnde->bcnde', givens[:, :, :, 1:].expand(-1, -1, -1, 2), th.cat((x[:, :, :, 1:], x[:, :, :, 0:1]), dim=-1))
        x_ref = x_ref_a + x_ref_b
        return x_ref.view((r.shape[0], r.shape[1], x.shape[1], -1))


def logmap0(y, c):
    sqrt_c = c ** 0.5
    y_norm = y.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    return y / y_norm / sqrt_c * artanh(sqrt_c * y_norm)
