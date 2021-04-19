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


def mobius_add(x, y, c):
    """Mobius addition of points in the Poincare ball with curvature c.

    Parameters
    ----------
    x: torch.Tensor
        size B x d with hyperbolic points
    y: torch.Tensor
        size B x d with hyperbolic points
    c: torch.Tensor
        size B x 1 with absolute hyperbolic curvatures

    Returns
    -------
    torch.Tensor
        shape B x d representing the element-wise Mobius addition of x and y.

    """
    x2 = th.sum(x * x, dim=-1, keepdim=True)
    y2 = th.sum(y * y, dim=-1, keepdim=True)
    xy = th.sum(x * y, dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / denom.clamp_min(MIN_NORM)


def project(x, c):
    """Project points to Poincare ball with curvature c.

    Parameters
    ----------
    x: torch.Tensor
        size B x d with hyperbolic points
    c: torch.Tensor
        size B x 1 with absolute hyperbolic curvatures

    Returns
    -------
    torch.Tensor
        projected hyperbolic points.

    """
    norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    eps = BALL_EPS[x.dtype]
    maxnorm = (1 - eps) / (c ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return th.where(cond, projected, x)


def expmap0(u, c):
    """Exponential map taken at the origin of the Poincare ball with curvature c.

    Parameters
    ----------
    u: torch.Tensor
        size B x d with hyperbolic points
    c: torch.Tensor
        size B x 1 with absolute hyperbolic curvatures

    Returns
    -------
    torch.Tensor
        tangent points.

    """
    sqrt_c = c ** 0.5
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return project(gamma_1, c)

def logmap0(y, c):
    """ Logarithmic map taken at the origin of the Poincare ball with curvature c.

    Parameters
    ----------
    y: torch.Tensor
        size B x d with tangent points
    c: torch.Tensor
        size B x 1 with absolute hyperbolic curvatures

    Returns
    -------
    torch.Tensor
        hyperbolic points.

    """
    sqrt_c = c ** 0.5
    y_norm = y.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    return y / y_norm / sqrt_c * artanh(sqrt_c * y_norm)
