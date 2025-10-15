import torch

def first_derivative(v, periodic=False):
    if periodic:
        return (v.roll(-1, dims=0) - v.roll(1, dims=0)) / 2.0
    else:
        return (v[2:] - v[:-2]) / 2.0

def second_derivative(v, periodic=False):
    if periodic:
        return v.roll(-1, dims=0) - 2 * v + v.roll(1, dims=0)
    else:
        return v[2:] - 2 * v[1:-1] + v[:-2]

def first_derivative_squared_norm(v, periodic=False):
    d1 = first_derivative(v, periodic=periodic)
    length = d1.shape[0]
    return torch.sum(d1 ** 2) / length

def second_derivative_squared_norm(v, periodic=False):
    d2 = second_derivative(v, periodic=periodic)
    length = d2.shape[0]
    return torch.sum(d2 ** 2) / length

def smooth_max(v, p=10, dim=None):
    '''Compute the (smooth) max of the values by computing their p-norm, p >> 1.'''
    return torch.linalg.norm(v, ord=p, dim=dim)

def unit_vectors(v):
    '''Compute the deviation from unit norm of the vectors in v, shape (n, 3) or (n, m, 3)'''
    assert v.shape[-1] == 3
    norms = torch.linalg.norm(v, dim=-1).flatten()
    return smooth_max((norms - 1.0) ** 2, dim=0)

def get_pairings_exact(N, shift=0):
    """
    Returns a tensor of shape (1, N, 2) representing exact pairings for N elements,
    optionally shifted cyclically by 'shift'.

    Each element i is paired with (i + shift) % N: [i, (i + shift) % N].

    Args:
        N (int): Number of elements to pair.
        shift (int, optional): Cyclic shift to apply. Default is 0.

    Returns:
        torch.Tensor: Pairings tensor with exact matches.
    """
    pairings = torch.zeros(size=(1, N, 2), dtype=torch.int64)
    pairings[0, :, 0] = torch.arange(N)
    pairings[0, :, 1] = torch.roll(torch.arange(N), shifts=shift, dims=0)
    return pairings

def get_pairings_all(N):
    """
    Returns a tensor of shape (N, N, 2) representing all cyclic pairings for N elements.
    For each shift, elements are paired with a cyclically shifted version of themselves.
    
    Args:
        N (int): Number of elements to pair.
    
    Returns:
        torch.Tensor: Pairings tensor with all cyclic matches.
    """
    pairings = torch.zeros(size=(N, N, 2), dtype=torch.int64)
    pairings[:, :, 0] = torch.arange(N)
    for id_pairing in range(N):
        pairings[id_pairing, :, 1] = torch.roll(torch.arange(N), shifts=id_pairing, dims=0)
    return pairings

def remove_useless_vertices(vertices, elements):
    '''Remove useless vertices depending on the unused IDs in the elements.
    
    Args:
        vertices (torch.Tensor): (n_vertices, 3) tensor of the vertices
        elements (torch.Tensor): (n_elements, n_vertices_per_element) tensor of the elements
        
    Returns:
        new_vertices (torch.Tensor): (n_new_vertices, 3) tensor of the vertices
        new_elements (torch.Tensor): (n_elements, n_vertices_per_element) tensor of the elements with the new indices
    '''

    unique_vertex_indices = torch.sort(torch.unique(elements)).values
    vertex_id_to_new = - torch.ones(size=(vertices.shape[0],), dtype=torch.int64)
    vertex_id_to_new[unique_vertex_indices] = torch.arange(unique_vertex_indices.shape[0])
    new_vertices = vertices[unique_vertex_indices]
    new_elements = vertex_id_to_new[elements]
    
    return new_vertices, new_elements

def write_polyline_to_obj(file, pts, write_edges=True, edge_idx_offset=0, mode='w', object_name=None, closed_curve=False):
    """
    Write a polyline to an OBJ file.

    Args:
    file (str): The path to the OBJ file.
    pts (torch.Tensor): A tensor of shape (n, 3) representing the points.
    write_edges (bool): Whether to write the edges of the polyline.
    mode (str): The mode in which to open the file. 'w' for writing (overwrite) and 'a' for appending.
    """
    with open(file, mode) as f:
        if object_name is not None:
            f.write(f"o {object_name}\n")
        for pt in pts:
            f.write(f"v {pt[0]} {pt[1]} {pt[2]}\n")
        if write_edges:
            n_pts = pts.shape[0]
            for i in range(edge_idx_offset, edge_idx_offset + n_pts - 1):
                f.write(f"l {i+1} {i+2}\n")
            if closed_curve:
                f.write(f"l {edge_idx_offset+n_pts} {edge_idx_offset+1}\n")

def write_mesh_to_obj(file, V, F, mode='w', object_name=None):
    """
    Write a mesh to an OBJ file.

    Args:
    file (str): The path to the OBJ file.
    V (torch.Tensor): A tensor of shape (n_vertices, 3) representing the vertices.
    F (torch.Tensor): A tensor of shape (n_faces, n_vertices_per_face) representing the faces.
    mode (str): The mode in which to open the file. 'w' for writing (overwrite) and 'a' for appending.
    """
    with open(file, mode) as file:
        if object_name is not None:
            file.write(f"o {object_name}\n")
        for v in V:
            file.write(f"v {v[0]} {v[1]} {v[2]}\n")
        if F.shape[1] == 3:
            for f in F:
                file.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")
        elif F.shape[1] == 4:
            for f in F:
                file.write(f"f {f[0]+1} {f[1]+1} {f[2]+1} {f[3]+1}\n")
        else:
            raise ValueError("Only triangles and quads are supported.")
            
def load_curve_from_obj(file, dtype=torch.float64):
    '''
    Args:
        file (str): The path to the OBJ file.
        dtype (torch.dtype): The data type of the vertices.
        
    Returns:
        vertices (torch.Tensor): (n_vertices, 3) tensor of the vertices
    '''
    with open(file, 'r') as f:
        lines = f.readlines()

    vertices = []
    vertices_curr_curve = []
    for line in lines:
        if line.startswith('v '):
            vertex = line.split(' ')[1:]
            vertices_curr_curve.append([float(vertex[0]), float(vertex[1]), float(vertex[2])])
        elif line.startswith('end') or line.startswith('l '):
            vertices_curr_curve = torch.tensor(vertices_curr_curve, dtype=dtype)
            vertices.append(vertices_curr_curve)
            vertices_curr_curve = []
        else:
            raise ValueError(f"Cannot parse line: {line}")
    if len(vertices) == 0:
        vertices = torch.tensor(vertices_curr_curve, dtype=dtype)
    elif len(vertices) == 1:
        vertices = vertices[0]
    else:
        raise ValueError("The OBJ file contains multiple curves. Use load_many_curves_from_obj instead.")

    return vertices

def load_many_curves_from_obj(file, dtype=torch.float64):
    '''
    Args:
        file (str): The path to the OBJ file.
        dtype (torch.dtype): The data type of the vertices.
        
    Returns:
        vertices (list[torch.Tensor]): A list of (n_vertices, 3) tensors of the vertices
    '''
    with open(file, 'r') as f:
        lines = f.readlines()

    vertices_curr_curve = []
    vertices = []
    for line in lines:
        if line.startswith('v '):
            vertex = line.split(' ')[1:]
            vertices_curr_curve.append([float(vertex[0]), float(vertex[1]), float(vertex[2])])
        if line.startswith('end') or line.startswith('l '):
            vertices_curr_curve = torch.tensor(vertices_curr_curve, dtype=dtype)
            vertices.append(vertices_curr_curve)
            vertices_curr_curve = []

    return vertices