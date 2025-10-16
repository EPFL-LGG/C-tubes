from Ctubes.geometry_utils import (
    rotate_about_axis, rotate_many_about_axis, rotate_about_axes, 
    compute_tangents, compute_plane_normals, parallel_transport,
    project_vector_on_plane_along_direction, rotation_matrix_from_vectors,
    is_tube_pq, angle_around_axis, align_point_cloud, project_vectors_on_planes_along_directions
)
from Ctubes.misc_utils import remove_useless_vertices
import numpy as np
import torch

PI = np.pi


def compute_ctube(X, G0, plane_normals=None, apex_loc_func=None, dihedral_angles=None, closed_curve=True):
    """
    Compute the mesh vertices of a C-tube defined by the directrix X and the generatrix G0.

    Args:
        X (torch.Tensor): Tensor of shape (M, 3) representing the directrix curve points in 3D space.
        G0 (torch.Tensor): Tensor of shape (N, 3) representing the initial cross-section (generatrix) vertices.
        plane_normals (torch.Tensor, optional): 
            - If shape (M, 3): Normals for each cross-section plane (planar profile).
            - If shape (M, N, 3): Normals for each vertex in each cross-section (non-planar profile).
            - If None: Computed automatically using 'bisecting' method.
        apex_loc_func (torch.Tensor, optional): Tensor of shape (M-1,) specifying the apex locating function for each segment. 
            If None, defaults to ones.
        dihedral_angles (torch.Tensor, optional): Tensor of shape (M-1, N) specifying dihedral angles for each cross-section edge.
            Only supported for planar profiles. Mutually exclusive with apex_loc_func.
        closed_curve (bool, optional): If True, asserts that the directrix curve is closed (first and last points are equal).

    Returns:
        torch.Tensor: Tensor of shape (M, N, 3) containing the computed mesh vertices of the C-tube, 
            where M is the number of directrix points and N is the number of generatrix vertices.
    """

    # --- Preprocessing ---

    if closed_curve:
        assert torch.allclose(X[0], X[-1]), "The curve is supposed to be closed, the first and last points must be the same."

    M = X.shape[0]
    N = G0.shape[0]
    tangents = compute_tangents(X)

    if plane_normals is None:
        plane_normals = compute_plane_normals(X, kind='bisecting', closed_curve=closed_curve)
        planar_profile = True
    elif isinstance(plane_normals, torch.Tensor) and len(plane_normals.shape) == 2:
        plane_normals_normalized = plane_normals / torch.linalg.norm(plane_normals, dim=1, keepdim=True)
        plane_normals = plane_normals_normalized
        planar_profile = True
    elif isinstance(plane_normals, torch.Tensor) and len(plane_normals.shape) == 3:
        assert plane_normals.shape[1] == N, "The list of plane normals must have the same length as the number of cross-section vertices."
        plane_normals_normalized = plane_normals / torch.linalg.norm(plane_normals, dim=2, keepdim=True)
        plane_normals = plane_normals_normalized
        planar_profile = False
    else:
        raise ValueError("The plane normals must be a tensor of either shape (M, 3) or (M, N, 3).")

    if apex_loc_func is None:
        apex_loc_func = torch.ones(M-1)

    if dihedral_angles is not None:
        assert planar_profile, "Using dihedral angles is only supported when the cross-sections are planar."
        assert dihedral_angles.shape == (M-1, N), "The dihedral angles must have shape (M-1, N)."
        assert apex_loc_func is None or torch.allclose(apex_loc_func, torch.tensor(1.0)), "Using dihedral angles or apex locating function is mutually exclusive."

    # --- Tube construction ---
    
    ctube_vertices = torch.zeros(size=(M, N, 3))

    if dihedral_angles is None:

        # Initialize the first cross-section
        ctube_vertices[0] = G0

        # Sweep the generatrix along the directrix
        for j in range(1, M):
            if planar_profile:
                op = X[j] + project_vector_on_plane_along_direction(X[j-1] - X[j], plane_normals[j], tangents[j-1])
                if j == 1:  # only for the first cross-section, we also need to back-project it to the first vertex's plane, as it was initialized perpendicular to the edge
                    ctube_vertices[0] = X[j-1].reshape(1, 3) + project_vectors_on_planes_along_directions(G0 - X[j-1].reshape(1, 3), plane_normals[0], -tangents[0].reshape(1, 3))
                ctube_vertices_prime = apex_loc_func[j-1] * (ctube_vertices[j-1] - X[j-1]) + op
            else:
                op = X[j]
                ctube_vertices_prime = apex_loc_func[j-1] * (ctube_vertices[j-1] - X[j-1]) + op
                if j == 1:  # only for the first cross-section, we also need to back-project it to the first vertex's plane, as it was initialized perpendicular to the edge
                    ctube_vertices[0] = X[j-1].reshape(1, 3) + project_vectors_on_planes_along_directions(G0 - X[j-1].reshape(1, 3), plane_normals[0], ctube_vertices_prime - G0)

            cs_vectors = ctube_vertices_prime - op.reshape(1, 3) if planar_profile else ctube_vertices[j-1] - X[j-1].reshape(1, 3) # shape (N, 3), the cross-section offsets to transport along the tube
            pn_tmp = plane_normals[j].reshape(1, 3) if planar_profile else plane_normals[j] # shape (1 or N, 3)
            ctube_vertices[j] = op.reshape(1, 3) + project_vectors_on_planes_along_directions(cs_vectors, pn_tmp, ctube_vertices_prime - ctube_vertices[j-1])

    else: # using dihedral_angles

        # Initialize the first cross-section (rotate it s.t. it lies in the plane defined by the first plane normal)
        rot = rotation_matrix_from_vectors(tangents[0], plane_normals[0])
        ctube_vertices[0, :, :] = torch.matmul(G0 - X[0], rot.T) + X[0]

        # Sweep the generatrix along the curve
        for j in range(1, M):

            # Compute the quad normals by rotating the plane normals about the cross-section edges
            e_cs = ctube_vertices[j-1, (torch.arange(N) + 1) % N] - ctube_vertices[j-1, :]
            e_cs = e_cs / torch.linalg.norm(e_cs, dim=1, keepdim=True)
            quad_normals = rotate_about_axes(plane_normals[j-1].reshape(1, 3).expand(N, -1), e_cs, dihedral_angles[j-1, :])

            # Compute the tube ridge directions
            tube_ridge_directions = torch.cross(torch.roll(quad_normals, shifts=1, dims=0), quad_normals, dim=1)
            tube_ridge_directions = tube_ridge_directions / torch.linalg.norm(tube_ridge_directions, dim=1, keepdim=True)

            # Project the cross-section vertices on the vertex planes along the tube ridge directions
            ctube_vertices[j] = X[j].reshape(1, 3) + project_vectors_on_planes_along_directions(ctube_vertices[j-1] - X[j].reshape(1, 3), plane_normals[j].reshape(1, 3), tube_ridge_directions)

    return ctube_vertices

def compute_parallel_transported_tube(directrix, theta, generatrix, init_scale=None, closed_curve=True):
    '''Computes the vertices of a tube around a curve defined by directrix using parallel transport.

    Args:
        directrix (torch.Tensor of shape (M, 3)): The points defining the curve.
        theta (float): The initial angle of the tube.
        generatrix (torch.Tensor of shape (N, 2)): The initial vertices (in the xy plane) of the tube to be swept along the curve.
        init_scale (float, optional): The initial scale factor of the cross-section on the first vertex plane. Default is 1.0.
        closed_curve (bool, optional): If True, the curve is assumed to be closed. The first and last points must be the same in this case.

    Returns:
        ctube_vertices (torch.Tensor of shape (M, N, 3)): The vertices of the tube.
    '''

    if closed_curve:
        assert torch.allclose(directrix[0], directrix[-1]), "The curve is supposed to be closed, the first and last points must be the same."

    M = directrix.shape[0]
    N = generatrix.shape[0]
    tangents = compute_tangents(directrix)

    plane_normals = compute_plane_normals(directrix, kind='bisecting', closed_curve=closed_curve)

    if init_scale is not None:
        # init_scale = torch.abs(torch.tensor(init_scale))  # use abs to avoid the need of setting bounds on init_scale
        # assert init_scale > 0.0, "The initial scale must be positive."
        if init_scale < 0.0:
            print("WARNING: negative init_scale {}!".format(init_scale))
    else:
        init_scale = 1.0

    # --- Initialization ---

    # Always initialize the cross-section perpendicular to the first edge's tangent.
    # This makes designing an initial cross-section more intuitive, as it can be directly designed 
    # in the xy-plane without worrying about the shearing induced by the projection on the vertex plane.
    # The actual initial and final cross-sections will then be projected onto the plane defined by the corresponding vertex tangent later.
    e1 = torch.cross(tangents[0], torch.tensor([0.0, 0.0, 1.0]), dim=0)
    e1 = e1 / torch.linalg.norm(e1)
    e2 = torch.cross(tangents[0], e1, dim=0)

    generatrix_scaled_inplane = torch.zeros(size=(N, 3))
    generatrix_scaled_inplane[:, :2] = generatrix * init_scale

    generatrix_rotated_inplane = rotate_about_axis(generatrix_scaled_inplane, torch.tensor([0.0, 0.0, 1.0]), theta)
    rot_xyz_to_frame = torch.stack([e1, e2, tangents[0]], dim=1)
    generatrix_oncurve = directrix[0].unsqueeze(0) + generatrix_rotated_inplane @ rot_xyz_to_frame.T   # perpendicular to the first edge's tangent

    # --- Tube construction ---
    
    ctube_vertices = torch.zeros(size=(M, N, 3))

    # Define vectors orthogonal to the tangent of the first edge pointing to the cross-section nodes
    ds_0 = generatrix_oncurve[:, :] - directrix[0]

    # Check that the vectors are orthogonal to the tangent
    assert torch.allclose(ds_0 @ tangents[0], torch.tensor(0.0))

    # Parallely transport the vectors along the curve
    ds_transported_on_edges = []  # per-edge, will have size (M-1, N, 3)
    ds_transported_on_edges.append(ds_0)
    for j in range(1, M-1):
        ds_prev = ds_transported_on_edges[j-1]
        ds_curr = parallel_transport(tangents[j-1].reshape(1, 3), tangents[j].reshape(1, 3), ds_prev)
        ds_transported_on_edges.append(ds_curr)
    ds_transported_on_edges = torch.cat(ds_transported_on_edges, dim=0).reshape(-1, N, 3)

    # Compute the cross-sections on vertices using the transported vectors on edges
    ctube_vertices = torch.zeros(size=(M, N, 3))

    # Back-project the first cross-section to the first vertex's plane
    ctube_vertices[0, :, :] = directrix[0].reshape(1, 3) + project_vectors_on_planes_along_directions(ds_transported_on_edges[0], plane_normals[0].reshape(1, 3), -tangents[0].reshape(1, 3))

    # Forward-project all the other cross-sections on the corresponding vertex's plane
    for j in range(1, M):
        ctube_vertices[j, :, :] = directrix[j].reshape(1, 3) + project_vectors_on_planes_along_directions(ds_transported_on_edges[j-1], plane_normals[j].reshape(1, 3), -tangents[j-1].reshape(1, 3))

    return ctube_vertices

# --------------------------------------------------------------------------------
# Strip unrolling (PQ strips)
# --------------------------------------------------------------------------------

def compute_ctube_topology(M, N):
    '''Computes the topology of the vertices for the tube from compute_ctube.
    
    Args:
        M (int): The number of discretization points of the curve.
        N (int): The number of vertices of the cross section.
        
    Returns:
        quads (torch.Tensor of shape (N*(M-1), 4)): The quads of the tube.
        end_faces (torch.Tensor of shape (2, N)): The faces at the ends of the tube.
    '''
    vid_cross_section0 = torch.arange(N)
    vid_cross_section1 = torch.arange(N, 2*N)
    quads01 = torch.stack([
        vid_cross_section0, 
        vid_cross_section1,
        torch.roll(vid_cross_section1, shifts=-1, dims=0),
        torch.roll(vid_cross_section0, shifts=-1, dims=0),
    ], dim=1)
    
    quads = torch.zeros(size=(M-1, N, 4), dtype=torch.int32)
    quads = torch.arange(0, M-1).reshape(-1, 1, 1) * N + quads01.unsqueeze(0)
    quads = quads.reshape(-1, 4)
    
    first_cs = torch.zeros(size=(N+1,), dtype=torch.int32)
    first_cs[:-1] = torch.arange(0, N)
    first_cs[-1] = 0
    last_cs = torch.zeros(size=(N+1,), dtype=torch.int32)
    last_cs[:-1] = torch.arange(N * (M - 1), N * M)
    last_cs[-1] = N * (M - 1)
    torch.arange(N)
    end_faces = torch.stack([
        first_cs,
        last_cs,
    ], dim=0)
    
    return quads, end_faces

def align_quad_xy(vertices_quad, reference_vertex, reference_edge):
    '''
    Args:
        vertices_quad (torch.Tensor): (4, 3) tensor of the vertices of the quad
        reference_vertex (torch.Tensor): (3,) tensor of the reference vertex: vertex[-1] will be aligned with this vertex
        reference_edge (torch.Tensor): (3,) tensor of the reference edge: it must be in the xy-plane! Edge from vertex[-1] to vertex[0] will be aligned with this edge
    
    Returns:
        vertices_quad_aligned (torch.Tensor): (4, 3) tensor of the vertices of the quad aligned
    '''
    # First align the reference vertex
    vertices_quad_translated = vertices_quad + (reference_vertex - vertices_quad[-1]).unsqueeze(0)
    
    # Then align the reference edge
    edge = vertices_quad_translated[0] - vertices_quad_translated[-1]
    edge = edge / torch.linalg.norm(edge) # no inplace operation
    reference_edge[2] = 0.0 # make sure the reference edge is in the xy-plane
    reference_edge = reference_edge / torch.linalg.norm(reference_edge)
    axis_rot = torch.cross(edge, reference_edge, dim=0)
    norm_axis = torch.linalg.norm(axis_rot)
    cos_alpha = edge @ reference_edge
    alpha = torch.atan2(norm_axis, cos_alpha)
    center_rot = vertices_quad_translated[-1].unsqueeze(0)
    vertices_quad_edge_aligned = rotate_about_axis(vertices_quad_translated-center_rot, axis_rot/norm_axis, alpha) + center_rot

    # Then align the normal to the z-axis
    normal = torch.cross(vertices_quad_edge_aligned[1] - vertices_quad_edge_aligned[0], vertices_quad_edge_aligned[2] - vertices_quad_edge_aligned[0], dim=0)
    normal = normal / torch.linalg.norm(normal)
    z_axis = torch.tensor([0.0, 0.0, 1.0])
    if torch.allclose(normal, z_axis, 1e-10):  # if already aligned with z-axis, no need to rotate
        vertices_quad_aligned = vertices_quad_edge_aligned
    elif torch.allclose(normal, -z_axis, 1e-10):  # if aligned with -z-axis, rotate 180 degrees about reference edge
        vertices_quad_aligned = rotate_about_axis(vertices_quad_edge_aligned-center_rot, reference_edge/norm_axis, PI) + center_rot
    else:
        axis_rot = torch.cross(normal, z_axis, dim=0)
        norm_axis = torch.linalg.norm(axis_rot)
        cos_alpha = normal[2]
        alpha = torch.atan2(norm_axis, cos_alpha)
        center_rot = vertices_quad_edge_aligned[0].unsqueeze(0)
        vertices_quad_aligned = rotate_about_axis(vertices_quad_edge_aligned-center_rot, axis_rot/norm_axis, alpha) + center_rot

    return vertices_quad_aligned

def get_flattened_strips(ctube_vertices, N, quads):
    '''Gets the vertices and quads of the strips of the tube.
    
    Args:
        ctube_vertices (torch.Tensor of shape (N*M, 3)): The vertices of the tube.
        N (int): The number of vertices of the cross section.
        quads (torch.Tensor of shape (3*(M-1), 4)): The quads of the tube.
        
    Returns:
        vertices_per_strip (list of torch.Tensor of shape (n_vertices_strip, 3)): The vertices of each strip.
        quads_per_strip (list of torch.Tensor of shape (n_quads_strip, 4)): The quads of each strip
        labels_per_strip (list of strings): the labels associated to each strip
    '''
    
    vertices_per_strip = []
    quads_per_strip = []
    labels_per_strip = ["{}".format(strip_id) for strip_id in range(N)]
    for strip_id in range(N):
        quads_strip = quads.reshape(-1, N, 4)[:, strip_id]
        vertices_strip = ctube_vertices[quads_strip]
        triangle_vertices_strip = ctube_vertices.clone()
        vertex_ref = torch.zeros(size=(3,))
        edge_ref = torch.tensor([1.0, 0.0, 0.0])

        for id_quad in range(quads_strip.shape[0]):
            vertices_quad_aligned = align_quad_xy(vertices_strip[id_quad], vertex_ref, edge_ref)
            triangle_vertices_strip[quads_strip[id_quad]] = vertices_quad_aligned
            vertex_ref = vertices_quad_aligned[2]
            edge_ref = vertices_quad_aligned[1] - vertices_quad_aligned[2]
            edge_ref /= torch.linalg.norm(edge_ref)
            
        new_vertices_strip, new_quads = remove_useless_vertices(triangle_vertices_strip, quads_strip)
        vertices_per_strip.append(new_vertices_strip)
        quads_per_strip.append(new_quads)

    return vertices_per_strip, quads_per_strip, labels_per_strip


def split_strip(vertices_strip, quads_strip, idx_split, labels_strip):
    '''Splits the strip into multiple strips.
    
    Args:
        vertices_strip (torch.Tensor of shape (n_vertices_strip, 3)): The vertices of the strip.
        quads_strip (torch.Tensor of shape (n_quads_strip, 4)): The quads of the strip.
        idx_split (torch.Tensor of shape (n_splits+1,)): The indices of the splits, operating at the quads level.
        labels_strip (string): The label of the strip.
        
    Returns:
        vertices_per_strip (list of torch.Tensor of shape (n_vertices_strip, 3)): The vertices of each strip.
        quads_per_strip (list of torch.Tensor of shape (n_quads_strip, 4)): The quads of each strip
        labels_per_strip (list of strings): the labels associated to each strip
    '''
    
    vertices_per_strip, quads_per_strip, labels_per_strip = [], [], []
    for id_strip in range(len(idx_split) - 1):
        v_tmp, q_tmp = remove_useless_vertices(vertices_strip, quads_strip[idx_split[id_strip]:idx_split[id_strip+1]])
        vertices_per_strip.append(v_tmp)
        quads_per_strip.append(q_tmp)
        labels_per_strip.append(labels_strip+"_{}".format(id_strip))
        
    return vertices_per_strip, quads_per_strip, labels_per_strip

def compute_unrolled_strips(ctube_vertices, triangulate=False, global_face_indices=False, y_offset_total=None, y_offset_per_strip=None, selected_strips=None, axes_first_edges=None, points_first_nodes=None):
    '''Get the unrolled strips and align them in the x-y plane.'''
    
    M = ctube_vertices.shape[0]
    N = ctube_vertices.shape[1]
    
    if is_tube_pq(ctube_vertices) and not triangulate:  # PQ-mesh, we can use quads
        faces, end_faces = compute_ctube_topology(M, N)
        vertices_per_strip, faces_per_strip, labels_per_strip = get_flattened_strips(ctube_vertices.reshape(-1, 3), N, faces)
    
    else: # non-PQ-mesh, we have to use triangles
        faces, end_faces = compute_ctube_topology_tri(M, N)
        vertices_per_strip, faces_per_strip, labels_per_strip = get_flattened_strips_tri(ctube_vertices.reshape(-1, 3), N, faces)

    if selected_strips is None:
        selected_strips = range(len(vertices_per_strip))  # select all
    vertices_per_strip = [vertices_per_strip[id_strip] for id_strip in selected_strips]
    faces_per_strip = [faces_per_strip[id_strip] for id_strip in selected_strips]
    labels_per_strip = [labels_per_strip[id_strip] for id_strip in selected_strips]

    # Align the bboxes of the strips along the x-axis, stack them along the y-axis to avoid overlaps
    curr_height = 0.0
    if y_offset_total is not None:
        curr_height += y_offset_total
    all_strip_vertices = []
    for strip_i, verts_tmp in enumerate(vertices_per_strip):

        # Orient the strips, either:
        if axes_first_edges is not None and points_first_nodes is not None:
            # A) by alinging the first edge with the provided axis
            axis_first_edge = axes_first_edges[strip_i]
            e01 = verts_tmp[1] - verts_tmp[0]
            e3 = torch.tensor([0.0, 0.0, 1.0])
            angle = torch.tensor(angle_around_axis(e3, e01, axis_first_edge))
            verts_tmp_aligned = rotate_about_axis(verts_tmp, e3, angle)
        else:
            # B) by SVD
            verts_tmp_aligned = align_point_cloud(verts_tmp, pure_rotation=True)

            # Always rotate s.t. the first edge tangent is pointing downwards 
            # (s.t. we can always read how to pair up the strips from the 2d plot)
            faces_tmp = faces_per_strip[strip_i]  # extract local quad indices
            side_edge = verts_tmp_aligned[faces_tmp[0, [0, -1]]]  # select the side edge
            if side_edge[0, 1] > side_edge[1, 1]:
                verts_tmp_aligned = rotate_about_axis(verts_tmp_aligned, torch.tensor([0.0, 0.0, 1.0]), np.pi)

        aabb_verts_tmp_aligned = torch.stack([verts_tmp_aligned.min(dim=0).values, verts_tmp_aligned.max(dim=0).values], dim=0)
        verts_tmp_aligned = verts_tmp_aligned - aabb_verts_tmp_aligned[0].unsqueeze(0)
        verts_tmp_aligned[:, 1] = verts_tmp_aligned[:, 1] + curr_height
        aabb_extent = aabb_verts_tmp_aligned[1] - aabb_verts_tmp_aligned[0]
        delta_height = 1.2 * aabb_extent[1].item() # 20% margin
        curr_height += delta_height

        if y_offset_per_strip is not None:
            curr_height += y_offset_per_strip

        if points_first_nodes is not None:
            # Translate the strip to the provided point
            verts_tmp_aligned = verts_tmp_aligned - verts_tmp_aligned[0] + points_first_nodes[strip_i]

        all_strip_vertices.append(verts_tmp_aligned.clone())
    vertices_per_strip = all_strip_vertices

    if global_face_indices:
        # Create sequential face indices by adding the number of vertices in the previous strips
        offset = 0
        for i in range(0, len(faces_per_strip)):
            faces_per_strip[i] += offset
            offset += vertices_per_strip[i].shape[0]

    return vertices_per_strip, faces_per_strip

# --------------------------------------------------------------------------------
# Strip unrolling (triangulated, not necessarily PQ)
# --------------------------------------------------------------------------------
# NOTE: there is substantial code duplication with the PQ version

def align_triangle_xy(vertices_tri, reference_vertex, reference_edge, flip_normal=False):
    '''
    Args:
        vertices_triangle (torch.Tensor): (3, 3) tensor of the vertices of the triangle
        reference_vertex (torch.Tensor): (3,) tensor of the reference vertex: vertex[0] will be aligned with this vertex
        reference_edge (torch.Tensor): (3,) tensor of the reference edge: it must be in the xy-plane! Edge from vertex[0] to vertex[1] will be aligned with this edge
        flip_normal (bool): If True, the normal of the triangle will be flipped
    
    Returns:
        vertices_triangle_aligned (torch.Tensor): (3, 3) tensor of the vertices of the triangle aligned
    '''
    # Align the reference vertex
    vertices_tri_translated = vertices_tri + (reference_vertex - vertices_tri[0]).unsqueeze(0)
    
    # Align the reference edge
    edge = vertices_tri_translated[1] - vertices_tri_translated[0]
    edge = edge / torch.linalg.norm(edge) # no inplace operation
    assert torch.abs(reference_edge[2]) < 1.0e-14, "The reference edge must be in the xy-plane"
    reference_edge = reference_edge / torch.linalg.norm(reference_edge)
    axis_rot = torch.cross(edge, reference_edge, dim=0)
    norm_axis = torch.linalg.norm(axis_rot)
    cos_alpha = edge @ reference_edge
    alpha = torch.atan2(norm_axis, cos_alpha)
    center_rot = vertices_tri_translated[0].unsqueeze(0)
    vertices_tri_edge_aligned = rotate_about_axis(vertices_tri_translated-center_rot, axis_rot/norm_axis, alpha) + center_rot
    
    # Align the normal to the z-axis
    normal = torch.cross(vertices_tri_edge_aligned[1] - vertices_tri_edge_aligned[0], vertices_tri_edge_aligned[2] - vertices_tri_edge_aligned[0], dim=0)
    if flip_normal:
        normal = -normal
    normal = normal / torch.linalg.norm(normal)
    axis_rot = torch.cross(normal, torch.tensor([0.0, 0.0, 1.0]), dim=0)
    norm_axis = torch.linalg.norm(axis_rot)
    cos_alpha = normal[2]
    alpha = torch.atan2(norm_axis, cos_alpha)
    center_rot = vertices_tri_edge_aligned[0].unsqueeze(0)
    vertices_tri_aligned = rotate_about_axis(vertices_tri_edge_aligned-center_rot, axis_rot/norm_axis, alpha) + center_rot
    
    return vertices_tri_aligned

def compute_ctube_topology_tri(M, N):
    '''Computes the topology of the vertices for the tube from compute_ctube.
    
    Args:
        M (int): The number of discretization points of the curve.
        N (int): The number of vertices of the cross section.
        
    Returns:
        triangles (torch.Tensor of shape (N*(M-1), 3)): The triangles making the tube.
        end_faces (torch.Tensor of shape (2, N)): The faces at the ends of the tube.
    '''
    vid_cross_section0 = torch.arange(N)
    vid_cross_section1 = torch.arange(N, 2*N)
    triangles01 = torch.zeros(size=(2*N, 3), dtype=torch.int32)
    triangles01[:, 0] = torch.cat((vid_cross_section0.roll(-1), vid_cross_section0))  # [1, 2, 0, 0, 1, 2] for N = 3
    triangles01[:, 1] = torch.cat([vid_cross_section0, vid_cross_section1.roll(-1)])  # [0, 1, 2, 4, 5, 3] for N = 3
    triangles01[:, 2] = torch.cat([vid_cross_section1.roll(-1), vid_cross_section1])  # [4, 5, 3, 3, 4, 5] for N = 3

    triangles = torch.zeros(size=(M-1, 2*N, 3), dtype=torch.int32)
    triangles = torch.arange(0, M-1).reshape(-1, 1, 1) * N + triangles01.unsqueeze(0)
    triangles = triangles.reshape(-1, 3)

    first_cs = torch.zeros(size=(N+1,), dtype=torch.int32)
    first_cs[:-1] = torch.arange(0, N)
    first_cs[-1] = 0
    last_cs = torch.zeros(size=(N+1,), dtype=torch.int32)
    last_cs[:-1] = torch.arange(N * (M - 1), N * M)
    last_cs[-1] = N * (M - 1)
    torch.arange(N)
    end_faces = torch.stack([
        first_cs,
        last_cs,
    ], dim=0)

    return triangles, end_faces

def get_flattened_strips_tri(ctube_vertices, N, triangles):
    '''Gets the vertices and triangles of the strips of the tube.
    
    Args:
        ctube_vertices (torch.Tensor of shape (N*M, 3)): The vertices of the tube.
        N (int): The number of vertices of the cross section.
        triangles (torch.Tensor of shape (2*N*(M-1), 3)): The triangles of the tube.
        
    Returns:
        vertices_per_strip (list of torch.Tensor of shape (n_vertices_strip, 3)): The vertices of each strip.
        triangles_per_strip (list of torch.Tensor of shape (n_triangles_strip, 3)): The triangles of each strip.
        labels_per_strip (list of strings): the labels associated to each strip
    '''
    from Ctubes.misc_utils import remove_useless_vertices
    
    vertices_per_strip = []
    triangles_per_strip = []
    labels_per_strip = ["{}".format(strip_id) for strip_id in range(N)]
    M = ctube_vertices.shape[0] // N
    for strip_id in range(N):
        strip_tris_idx = torch.arange(0, 2*(M-1)*N, N) + strip_id
        triangles_strip = triangles[strip_tris_idx]
        vertices_strip = ctube_vertices[triangles_strip]
        triangle_vertices_strip = torch.zeros_like(ctube_vertices)
        vertex_ref = torch.zeros(size=(3,))
        edge_ref = torch.tensor([1.0, 0.0, 0.0])

        for id_tri in range(triangles_strip.shape[0]):
            vertices_tri = vertices_strip[id_tri]
            flip_normal = (id_tri % 2 == 1)   # the triangle vertices are alternatively defined as cw or ccw
            vertices_tri_aligned = align_triangle_xy(vertices_tri, vertex_ref, edge_ref, flip_normal=flip_normal)

            # DEBUG: check isometric deformation
            from Ctubes.geometry_utils import triangle_area
            assert (torch.linalg.norm(vertices_tri[0] - vertices_tri[1]) - torch.linalg.norm(vertices_tri_aligned[0] - vertices_tri_aligned[1])) < 1.0e-12
            assert (torch.linalg.norm(vertices_tri[1] - vertices_tri[2]) - torch.linalg.norm(vertices_tri_aligned[1] - vertices_tri_aligned[2])) < 1.0e-12
            assert (torch.linalg.norm(vertices_tri[2] - vertices_tri[0]) - torch.linalg.norm(vertices_tri_aligned[2] - vertices_tri_aligned[0])) < 1.0e-12
            assert triangle_area(vertices_tri[0], vertices_tri[1], vertices_tri[2]) - triangle_area(vertices_tri_aligned[0], vertices_tri_aligned[1], vertices_tri_aligned[2]) < 1.0e-14
            
            # DEBUG: Check that vertices 0 and 1 of the current triangle are the same as vertices 1 and 2 of the previous triangle 
            if id_tri > 0:
                assert (vertices_tri_aligned[1] - triangle_vertices_strip[triangles_strip[id_tri-1]][2]).norm() < 1.0e-12
                assert (vertices_tri_aligned[0] - triangle_vertices_strip[triangles_strip[id_tri-1]][1]).norm() < 1.0e-12

            triangle_vertices_strip[triangles_strip[id_tri]] = vertices_tri_aligned
            vertex_ref = vertices_tri_aligned[1]
            edge_ref = vertices_tri_aligned[2] - vertices_tri_aligned[1]
            edge_ref /= torch.linalg.norm(edge_ref)

        new_vertices_strip, new_triangles = remove_useless_vertices(triangle_vertices_strip, triangles_strip)
        vertices_per_strip.append(new_vertices_strip)
        triangles_per_strip.append(new_triangles)

    return vertices_per_strip, triangles_per_strip, labels_per_strip

# --------------------------------------------------------------------------------
# Flaps
# --------------------------------------------------------------------------------
# NOTE: only PQ strips are currently supported – no triangulated strips

def create_flap_geometry(length, width, angle):
    c = torch.cos(angle)
    s = torch.sin(angle)
    
    flap_geometry = torch.zeros(size=(4, 3))
    flap_geometry[1] = torch.tensor([width * c, - width * s, 0.0])
    flap_geometry[2] = torch.tensor([length - width * c, - width * s, 0.0])
    flap_geometry[3] = torch.tensor([length, 0.0, 0.0])
    
    return flap_geometry

vmap_create_flap_geometry = torch.vmap(create_flap_geometry, in_dims=(0, 0, 0))

def create_flap_geometries(lengths, width, angle):
    c = torch.cos(angle)
    s = torch.sin(angle)
    
    flap_geometries = torch.zeros(size=(lengths.shape[0], 4, 3))
    flap_geometries[:, 1, 0] = width * c
    flap_geometries[:, 1, 1] = - width * s
    flap_geometries[:, 2, 0] = lengths - width * c
    flap_geometries[:, 2, 1] = - width * s
    flap_geometries[:, 3, 0] = lengths

    return flap_geometries

def add_flaps_to_strips(vertices_per_strip, quads_per_strip, width, angle, keep_flap_every=1):
    '''Adds flaps to the strips of the tube.
    '''
    
    vertices_with_flaps_per_strip = []
    polylines_with_flaps_per_strip = []
    edges_fold_per_strip = []
    
    for id_strip in range(len(vertices_per_strip)):
        quads_strip = quads_per_strip[id_strip]
        vertices_strip = vertices_per_strip[id_strip]
        vertices_quads_strip = vertices_strip[quads_strip]
        vertices_with_flaps = torch.zeros(size=(3 * (quads_strip.shape[0] + 1) + 1 + quads_strip.shape[0], 3))
        
        dir_edges = torch.zeros(size=(quads_strip.shape[0]+1, 3))
        dir_edges[:-1] = vertices_quads_strip[:, 1] - vertices_quads_strip[:, 0]
        dir_edges[-1] = vertices_quads_strip[-1, 2] - vertices_quads_strip[-1, 1]
        lengths_edges = torch.linalg.norm(dir_edges, dim=1)
        angles_edges = torch.atan2(dir_edges[:, 1], dir_edges[:, 0])
        
        flaps = create_flap_geometries(lengths_edges, width, angle)
        axes_rot = torch.zeros(size=(flaps.shape[0], 3))
        axes_rot[:, 2] = 1.0
        corner_ref = torch.zeros(size=(quads_strip.shape[0]+1, 3))
        corner_ref[:-1] = vertices_quads_strip[:, 0]
        corner_ref[-1] = vertices_quads_strip[-1, 1]
        flaps_registered = rotate_many_about_axis(flaps, axes_rot, angles_edges) + corner_ref.unsqueeze(1)
        flaps_registered = flaps_registered.reshape(-1, 3)
        
        id_keep = torch.arange(4 * (quads_strip.shape[0] + 1)).reshape(-1, 4)[:, :3].reshape(-1,)
        vertices_with_flaps[:3 * (quads_strip.shape[0] + 1)] = flaps_registered[id_keep]
        vertices_with_flaps[3 * (quads_strip.shape[0] + 1)] = flaps_registered[-1]
        vertices_with_flaps[3 * (quads_strip.shape[0] + 1) + 1:] = torch.flip(vertices_quads_strip[:, 3], dims=[0])

        id_keep_right = []
        edges_fold_quads_tmp = []
        edges_fold_flaps_tmp = []
        id_vertex_new = 0
        for id_quad in range(quads_strip.shape[0]):
            if id_quad % keep_flap_every == 0:
                edges_fold_quads_tmp.append(torch.tensor([id_vertex_new+3, -id_quad-2], dtype=torch.int32))
                id_keep_right += list(range(3*id_quad, 3*(id_quad+1)))
                edges_fold_flaps_tmp.append(torch.tensor([id_vertex_new, id_vertex_new+3], dtype=torch.int32))
                id_vertex_new += 3
            else:
                edges_fold_quads_tmp.append(torch.tensor([id_vertex_new+1, -id_quad-2], dtype=torch.int32))
                id_keep_right += [3*id_quad]
                id_vertex_new += 1
        id_keep_up = list(range(3 * quads_strip.shape[0], 3 * (quads_strip.shape[0] + 1) + 1))
        id_keep_left = list(range(3 * (quads_strip.shape[0] + 1) + 1, 3 * (quads_strip.shape[0] + 1) + 1 + quads_strip.shape[0]))
        id_filter = id_keep_right + id_keep_up + id_keep_left
        vertices_with_flaps = vertices_with_flaps[id_filter]
        
        edges_fold_flaps_tmp = torch.stack(edges_fold_flaps_tmp, dim=0).reshape(-1, 2)
        edges_fold_quads_tmp = torch.stack(edges_fold_quads_tmp, dim=0).reshape(-1, 2)
        edges_fold_quads_tmp[:, 1] += vertices_with_flaps.shape[0]
        edges_fold_tmp = torch.cat([edges_fold_flaps_tmp, edges_fold_quads_tmp], dim=0)

        polyline_tmp = torch.zeros(size=(vertices_with_flaps.shape[0]+1,), dtype=torch.int32)
        polyline_tmp[:-1] = torch.arange(vertices_with_flaps.shape[0]) 

        vertices_with_flaps_per_strip.append(vertices_with_flaps)
        polylines_with_flaps_per_strip.append(polyline_tmp)
        edges_fold_per_strip.append(edges_fold_tmp)
        

    return vertices_with_flaps_per_strip, polylines_with_flaps_per_strip, edges_fold_per_strip
