import torch
from Ctubes.geometry_utils import (
    compute_tangents, point_mesh_squared_distance, point_cloud_is_planar, compute_quad_normals,
    project_vectors_on_planes_along_directions, compute_cross_section_radii, compute_quad_centers, vmap_compute_curvature_angles,
    triangle_normal, compute_min_curve_self_distance, symmetric_chamfer_distance_squared
)
from Ctubes.misc_utils import second_derivative_squared_norm, smooth_max, unit_vectors

# ---------- Directrix ----------

def preserve_curve(vertices, curve):
    '''Preserves the curve by minimizing the chamfer distance between the tube and the curve.
    
    Args:
        vertices (torch.Tensor of shape (n_vertices, 3)): The vertices of the tube.
        curve (torch.Tensor of shape (n_curve, 3)): The curve to preserve.
    
    Returns:
        obj (torch.Tensor of shape (,)): The chamfer distance between the tube and the curve.
    '''
    return 0.5 * symmetric_chamfer_distance_squared(vertices, curve)

def min_directrix_self_distance(directrix, target_min_dist, closed_curve=False):
    M = directrix.shape[0]
    L_crv = torch.sum(torch.linalg.norm(directrix[1:] - directrix[:-1], dim=1))
    point_neighborhood_dist = 0.1 * L_crv
    min_self_distances = compute_min_curve_self_distance(directrix, neighborhood_arclength_dist=point_neighborhood_dist, closed=closed_curve)
    clamped_diffs = torch.min(min_self_distances - target_min_dist, torch.zeros(M))
    obj = torch.max(clamped_diffs ** 2)
    return obj

# ---------- Cross-Section ----------

def join_ends(vertices, N, pairings, transform=lambda x: x):
    '''Closes the tube by connecting the first and last vertices.
    
    Args:
        vertices (torch.Tensor of shape (n_vertices, 3)): The vertices of the tube.
        N (int): The number of vertices of the cross section.
        pairings (torch.Tensor of shape (n_pairings, n_pairs, 2)): The indices of the start vertices.
        transform (callable): The transformation to apply to the vertices of the last face before computing the chamfer distance with the first face. By default, no transformation is applied. Can be used e.g. to align a building block of structures with symmetry. 
    
    Returns:
        obj (torch.Tensor of shape (,)): The mismatch between pairs of vertices among all possible pairings
    '''
    face_start = vertices[:N]
    face_end = transform(vertices[-N:])
    distance_per_point_pair = torch.sum((face_end[pairings[..., 1]] - face_start[pairings[..., 0]]) ** 2, dim=-1)  # (n_pairings, n_pairs)
    objs = 0.5 * torch.mean(distance_per_point_pair, dim=-1)  # (n_pairings,)
    return torch.min(objs)

def match_target_cross_sections(vertices, cross_section_indices, target_cross_sections, pairings, project_on_target_plane=False):
    '''Match a set of target cross sections to the corresponding cross sections of the tube. 
    M is the total number of cross sections. 
    n_matches is the number of cross sections to match to the targets. 
    N is the number of vertices of the cross section. 
    n_pairings is the number of different pairings of vertices allowed.

    Args:
        vertices (torch.Tensor of shape (M, N, 3): The vertices of the tube.
        cross_section_indices (torch.Tensor of shape (n_matches)): The indices of the cross sections along the tube to be matched to the targets.
        target_cross_sections (torch.Tensor of shape (n_matches, N, 3)): The target cross sections.
        pairings (torch.Tensor of shape (n_matches, n_pairings, N, 2)): The indices identifying the pairs of vertices to match on the tube and target cross sections, see target_cross_sections.py for some example of pairings.
        project_on_target_plane (bool): Whether to measure the cross section mismatch on the plane of the target cross sections (True) or in 3D (False).

    Returns:
        obj (torch.Tensor of shape (,)): The mismatch between pairs of vertices among all possible pairings
    '''
    assert len(cross_section_indices) == len(target_cross_sections), "The number of cross sections {} and target cross sections {} must be the same.".format(len(cross_section_indices), len(target_cross_sections))
    
    faces = vertices[cross_section_indices]  # (n_matches, N, 3)
    target_faces = target_cross_sections  # (n_matches, N, 3)
    
    # Gather the paired vertices
    face_pairs = faces[:, pairings[..., 0]]  # (n_matches, n_pairings, N, 3)
    target_face_pairs = target_faces[:, pairings[..., 1]]  # (n_matches, n_pairings, N, 3)

    if project_on_target_plane:
        # Project the cross sections on the target planes
        assert [point_cloud_is_planar(target_faces[i]) for i in range(len(target_faces))], "The target cross sections are not planar, cannot project on the target planes."
        target_normals = torch.stack([triangle_normal(target_faces[i][0], target_faces[i][1], target_faces[i][2]) for i in range(len(target_faces))], dim=0)  # (n_matches, 3)
        target_normals = target_normals / torch.linalg.norm(target_normals, dim=1, keepdim=True)  # (n_matches, 3)
        target_normals = target_normals.unsqueeze(1).unsqueeze(1)  # unsqueeze N and n_pairings -> (n_matches, 1, 1, 3)
        centroids = torch.stack([torch.mean(target_faces[i], dim=0) for i in range(len(target_faces))], dim=0)  # (n_matches, 3)
        centroids = centroids.unsqueeze(1).unsqueeze(1)  # unsqueeze N and n_pairings -> (n_matches, 1, 1, 3)
        face_pairs = project_vectors_on_planes_along_directions(face_pairs - centroids, target_normals, target_normals) + centroids
    
    # Compute the squared differences
    diffs_sq = torch.sum((target_face_pairs - face_pairs) ** 2, dim=-1)  # (n_matches, n_pairings, N)
    
    # Compute the mean squared differences for each pairing
    obj_curr_cs_all_pairings = torch.mean(diffs_sq, dim=-1)  # (n_matches, n_pairings)
    
    # Find the minimum mismatch for each cross section
    obj_per_cs = torch.min(obj_curr_cs_all_pairings, dim=-1).values  # (n_matches)

    return smooth_max(obj_per_cs)

def match_target_cross_section_radius(directrix, ctube_vertices, target_radii):
    '''Match the radii of the swept cross-sections to the target radii.'''
    M = ctube_vertices.shape[0]
    assert target_radii.shape[0] == M, "The number of cross sections must be the same for the swept cross-sections and the target radii."
    radii = compute_cross_section_radii(directrix, ctube_vertices, on_normal_plane=True)  # shape (M, N)
    return 0.5 * torch.sum((radii - target_radii) ** 2) / M

# ---------- Alignment ----------

def constrain_tube_ridges_to_surface(ctube_vertices, V_surf, F_surf, ridge_indices):
    '''
    Constrain one of the ridges of the tube to lie on the surface.
    Project the ridge vertices onto the surface and compute the distance to the surface.
    '''
    obj = 0
    for i in ridge_indices:
        obj += point_mesh_squared_distance(ctube_vertices[:, i], V_surf, F_surf)
    return obj

def quad_tangency(vertices, quad_pair_tube_indices, quad_pair_disc_indices, quad_pair_cross_section_indices):
    '''Penalize deviations in the quad normals.

    Args:
        vertices (list of torch Tensors of shape (M, N, 3)): The vertices of the tubes, one tensor per tube.
        quad_pair_tube_indices (torch.Tensor of shape (n_pairs, 2)): The indices of the tubes forming the quad pair.
        quad_pair_disc_indices (torch.Tensor of shape (n_pairs, 2)): The indices of the directrix points forming the quad pair.
        quad_pair_cross_section_indices (torch.Tensor of shape (n_pairs, 2)): The indices of the cross-sections forming the quad pair.   
    '''
    assert quad_pair_tube_indices.shape == quad_pair_disc_indices.shape == quad_pair_cross_section_indices.shape, "The indices must have the same shape but are {} {} {}.".format(quad_pair_tube_indices.shape, quad_pair_disc_indices.shape, quad_pair_cross_section_indices.shape)
    n_pairs = quad_pair_tube_indices.shape[0]

    # Flatten the indices (even indices are the first quad of the pair, odd indices are the second quad of the pair)
    quad_pair_tube_indices_flat = quad_pair_tube_indices.flatten()
    quad_pair_disc_indices_flat = quad_pair_disc_indices.flatten()
    quad_pair_cross_section_indices_flat = quad_pair_cross_section_indices.flatten()

    # Compute the normals of the quads
    normals = compute_quad_normals(vertices, quad_pair_tube_indices_flat, quad_pair_disc_indices_flat, quad_pair_cross_section_indices_flat)
    
    # Compute the objective
    n1 = normals[::2]
    n2 = normals[1::2]
    cos_angles = torch.sum(n1 * n2, dim=1)
    obj = torch.sum((cos_angles + 1.0) ** 2)  # we want the two normals to be pointing in opposite directions

    return obj / n_pairs

def quad_orientation(vertices, quad_tube_indices, quad_disc_indices, quad_cross_section_indices, quad_orientations):
    '''Penalize deviations of the quad normals from the target orientations.

    Args:
        vertices (list of torch Tensors of shape (M, N, 3)): The vertices of the tubes, one tensor per tube.
        quad_tube_indices (torch.Tensor of shape (n_quads,)): The indices of the tubes forming the quad.
        quad_disc_indices (torch.Tensor of shape (n_quads,)): The indices of the quad.
        quad_cross_section_indices (torch.Tensor of shape (n_quads,)): The indices of the cross-sections.
        quad_orientations (torch.Tensor of shape (n_quads, 3)): The orientations of the quads.
    '''
    assert quad_tube_indices.shape[0] == quad_disc_indices.shape[0] == quad_cross_section_indices.shape[0] == quad_orientations.shape[0], "The indices and orientations must have the same shape."

    # Compute the normals of the quads
    normals = compute_quad_normals(vertices, quad_tube_indices, quad_disc_indices, quad_cross_section_indices)

    # Compute the objective
    cos_angle = torch.sum(normals * quad_orientations, dim=1)
    obj = torch.sum((cos_angle - 1.0) ** 2)  # we want the two normals to be pointing in the same direction

    return obj
    
def quad_tangency_any_pairing(vertices, directrix, quad_pair_tube_indices, quad_pair_disc_indices):
    '''Penalize deviations in the quad normals.
    Note: no need of quad_pair_cross_section_indices as all the possible pairings are tested.

    Args:
        vertices (list of torch Tensors of shape (M, N, 3)): The vertices of the tubes, one tensor per tube.
        directrix (list of torch Tensors of shape (M, 3)): The directrix points.
        quad_pair_tube_indices (torch.Tensor of shape (n_pairs, 2)): The indices of the tubes forming the quad pair.
        quad_pair_disc_indices (torch.Tensor of shape (n_pairs, 2)): The indices of the directrix points forming the quad pair.
    '''
    assert quad_pair_tube_indices.shape == quad_pair_disc_indices.shape, "The indices must have the same shape but are {} {}.".format(quad_pair_tube_indices.shape, quad_pair_disc_indices.shape)
    assert len(vertices) == len(directrix), "The number of tubes and the number of curves must be the same but are {} {}.".format(len(vertices), len(directrix))
    n_pairs = quad_pair_tube_indices.shape[0]

    obj = torch.tensor(0.0)

    # Compute the edge midpoints as the average of the corresponding curve points
    edge_midpoints = [(pts[1:] + pts[:-1]) / 2.0 for pts in directrix]

    for i in range(n_pairs):
        # Compute the direction connecting the edge midpoints (from the first to the second tube)
        t0 = quad_pair_tube_indices[i, 0]
        t1 = quad_pair_tube_indices[i, 1]
        d0 = quad_pair_disc_indices[i, 0]
        d1 = quad_pair_disc_indices[i, 1]
        ncs0 = vertices[t0].shape[1]
        ncs1 = vertices[t1].shape[1]
        align_direction = edge_midpoints[t1][d1] - edge_midpoints[t0][d0]
        align_direction = align_direction / torch.linalg.norm(align_direction)

        # Compute all the normals of the quads
        normals0 = compute_quad_normals(vertices, torch.full((ncs0,), t0, dtype=torch.int64), torch.full((ncs0,), d0, dtype=torch.int64), torch.arange(ncs0))
        normals1 = compute_quad_normals(vertices, torch.full((ncs0,), t1, dtype=torch.int64), torch.full((ncs0,), d1, dtype=torch.int64), torch.arange(ncs1))

        # Compute the dot product between the normals and the edge direction.
        # For the first tube, select the normal that is the best aligned with the alignmnent direction.
        # For the second tube, select the normal that is the best aligned with minus the alignment direction.
        cos_angles0 = torch.sum(normals0 * align_direction, dim=1)
        cos_angles1 = torch.sum(normals1 * -align_direction, dim=1)

        # Select the best alignment for each cross section
        best_alignment0 = torch.argmax(cos_angles0)
        best_alignment1 = torch.argmax(cos_angles1)

        # Compute the objective using the best alignment
        obj += (cos_angles0[best_alignment0] - 1.0) ** 2 + (cos_angles1[best_alignment1] - 1.0) ** 2

    return obj / n_pairs

def quad_distance(vertices, quad_pair_tube_indices, quad_pair_disc_indices, quad_pair_cross_section_indices, target_distances):
    '''Penalize deviations of the distances between the quads from the target distances.
    The distance is computed as the signed distance between the centroids of the quads along the direction given by the .

    Args:
        vertices (list of torch Tensors of shape (M, N, 3)): The vertices of the tubes, one tensor per tube.
        quad_pair_tube_indices (torch.Tensor of shape (n_pairs, 2)): The indices of the tubes forming the quad pair.
        quad_pair_disc_indices (torch.Tensor of shape (n_pairs, 2)): The indices of the directrix points forming the quad pair.
        quad_pair_cross_section_indices (torch.Tensor of shape (n_pairs, 2)): The indices of the cross-sections forming the quad pair.   
        target_distances (torch.Tensor of shape (n_pairs,)): The target distances between the quads.
    '''
    assert quad_pair_tube_indices.shape[0] == quad_pair_disc_indices.shape[0] == quad_pair_cross_section_indices.shape[0] == target_distances.shape[0], "The indices and target distances must have the same shape."

    # Flatten the indices (even indices are the first quad of the pair, odd indices are the second quad of the pair)
    quad_pair_tube_indices_flat = quad_pair_tube_indices.flatten()
    quad_pair_disc_indices_flat = quad_pair_disc_indices.flatten()
    quad_pair_cross_section_indices_flat = quad_pair_cross_section_indices.flatten()
    
    # Compute the centroids of the quads
    centroids = compute_quad_centers(vertices, quad_pair_tube_indices_flat, quad_pair_disc_indices_flat, quad_pair_cross_section_indices_flat)
    
    # Compute normals
    normals = compute_quad_normals(vertices, quad_pair_tube_indices_flat, quad_pair_disc_indices_flat, quad_pair_cross_section_indices_flat)

    # Compute the objective
    centroid_vecs = centroids[::2] - centroids[1::2]
    distances = torch.linalg.norm(centroid_vecs, dim=1)
    dot_products = torch.sum(centroid_vecs * normals[1::2], dim=1)
    signs = torch.tanh(10.0 * dot_products)  # smooth sign approximation
    obj = torch.sum((signs * distances - target_distances) ** 2)

    return obj

# ---------- Regularization ----------

def smooth_plane_normal_diffs(plane_normals, closed_curve=False):
    '''Smooth the plane normals.'''
    return 0.5 * second_derivative_squared_norm(plane_normals, periodic=closed_curve)

def smooth_apex_loc_func(apex_loc_func, closed_curve=False):
    '''Smooth the apex location.'''
    return 0.5 * second_derivative_squared_norm(apex_loc_func, periodic=closed_curve)

def smooth_tube_ridges(ctube_vertices, closed_curve=False):
    '''Smooth the gluing curves of the tube.'''
    return 0.5 * second_derivative_squared_norm(ctube_vertices, periodic=closed_curve)

def preserve_tube_ridge_edge_directions(ctube_vertices, directrix, min_length=0.0):
    '''Penalize flipping in the tube ridge (which would result in self-intersections and quad inversion).'''
    tangents = compute_tangents(directrix)
    N = ctube_vertices.shape[1]
    tangents_repeated = tangents.unsqueeze(1).repeat(1, N, 1) # (M-1, N, 3)
    ridge_edges = ctube_vertices[1:, :, :] - ctube_vertices[:-1, :, :] # (M-1, N, 3)
    dot_products = torch.einsum('ijk,ijk->ij', tangents_repeated, ridge_edges) # (M-1, N)
    ridge_edge_lengths = torch.linalg.norm(ridge_edges, dim=2) # (M-1, N)
    signs = torch.sign(dot_products)
    signed_ridge_edge_lengths = signs * ridge_edge_lengths

    val_per_ridge_edge = torch.where(dot_products > min_length, torch.tensor(0, dtype=dot_products.dtype), (signed_ridge_edge_lengths - min_length) ** 2)  # elementwise 0 if x > min_length, (x - min_length)^2 otherwise
    max_val_per_ridge = smooth_max(val_per_ridge_edge, dim=0)  # max over each gluing curve
    max_val = smooth_max(max_val_per_ridge, dim=0) # max over the cross-section vertices
    return max_val

def unitary_plane_normals(plane_normals):
    '''Constrain the plane normals to be unitary.'''
    return unit_vectors(plane_normals)
