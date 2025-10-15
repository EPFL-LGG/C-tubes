import torch

def helix_parameterization(t, r, h):
    '''
    Args:
        t: torch.Tensor of shape (n,)
        r: float
        h: float
    
    Returns:
        torch.Tensor of shape (n, 3)
    '''
    return torch.stack([
        r * torch.cos(t), 
        r * torch.sin(t), 
        h * t,
    ], dim=-1)

def trefoil_parameterization(t):
    '''
    Args:
        t: torch.Tensor of shape (n,)
    
    Returns:
        torch.Tensor of shape (n, 3)
    '''
    return torch.stack([
        torch.sin(t) + 2.0 * torch.sin(2.0 * t),
        torch.cos(t) - 2.0 * torch.cos(2.0 * t),
        -torch.sin(3.0 * t),
    ], dim=-1)
    
def figure_eight_parameterization(t):
    '''
    Args:
        t: torch.Tensor of shape (n,)
    
    Returns:
        torch.Tensor of shape (n, 3)
    '''
    return torch.stack([
        (2.0 + torch.cos(2.0 * t)) * torch.cos(3.0 * t),
        (2.0 + torch.cos(2.0 * t)) * torch.sin(3.0 * t),
        torch.sin(2.0 * t),
    ], dim=-1)
    
def kfoil_parameterization(t, k):
    '''
    Args:
        t: torch.Tensor of shape (n,)
        k: int, 2k+1 gives the number of lobes: k=1 is trefoil, k=2 is cinquefoil, etc
        
    Returns:
        torch.Tensor of shape (n, 3)
    '''
    return torch.stack([
        (2.0 + torch.cos(2.0 * t / (2.0 * k + 1))) * torch.cos(t),
        (2.0 + torch.cos(2.0 * t / (2.0 * k + 1))) * torch.sin(t),
        torch.sin(2.0 * t / (2.0 * k + 1)),
    ], dim=-1)

def torus_knot_parameterization(t, p, q, r, R):
    '''
    Args:
        t: torch.Tensor of shape (n,)
        p: int
        q: int
        r: float
        R: float

    Returns:
        torch.Tensor of shape (n, 3)
    '''

    x = (R + r * torch.cos(q * t)) * torch.cos(p * t)
    y = (R + r * torch.cos(q * t)) * torch.sin(p * t)
    z = r * torch.sin(q * t)
    return torch.stack([x, y, z], dim=-1)

def conical_spiral_parametrization(t, m=1, r=None):
    '''
    Args:
        t: torch.Tensor of shape (n,)
    
    Returns:
        torch.Tensor of shape (n, 3)
    '''
    if r is None:
        r = t
    return torch.stack([
        r * torch.cos(t),
        r * torch.sin(t),
        m*r,
    ], dim=-1)

def torus_knot_parameterization_maekawa_scholz(t, p, q, r, R):
    '''
    Follow parametrization from [Maekawa and Scholz 2024]

    Args:
        t: torch.Tensor of shape (n,)
        p: int
        q: int
        r: float
        R: float

    Returns:
        torch.Tensor of shape (n, 3)
    '''

    x = (R + r * torch.cos(q * t)) * torch.cos(p * t + torch.pi/q)
    y = (R + r * torch.cos(q * t)) * torch.sin(p * t + torch.pi/q)
    z = r * torch.sin(q * t)
    return torch.stack([x, y, z], dim=-1)

def circular_arc_in_yz_plane(t, r=None, z_scale=1):
    '''
    Args:
        t: torch.Tensor of shape (n,)
    
    Returns:
        torch.Tensor of shape (n, 3)
    '''
    if r is None:
        r = t
    return torch.stack([
        torch.zeros_like(t),
        r * torch.cos(t),
        r * torch.sin(t) * z_scale,
    ], dim=-1)