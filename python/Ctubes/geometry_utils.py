import torch
import torch.nn.functional as F
from Ctubes.misc_utils import smooth_max

def rotate_about_axis(vectors, axis, angle):
    '''Rotates vectors about an axis by an angle.
    Args:
        vectors (torch.Tensor of shape (n_vectors, 3)): The vectors to rotate.
        axis (torch.Tensor of shape (3,)): The axis of rotation: must be normalized!!
        angle (torch.Tensor of shape (,)): The angle of rotation, using the right-hand rule.
        
    Returns:
        vectors_rotated torch.Tensor of shape (n_vectors, 3): The rotated vectors.
    '''
    if isinstance(angle, float):
        angle = torch.tensor(angle)
    vectors_rotated = vectors * torch.cos(angle) + torch.cross(axis.unsqueeze(0), vectors, dim=-1) * torch.sin(angle) + axis.unsqueeze(0) * (vectors @ axis).unsqueeze(-1) * (1.0 - torch.cos(angle))
    return vectors_rotated

rotate_many_about_same_axis = torch.vmap(rotate_about_axis, in_dims=(0, None, 0))
rotate_many_about_axis = torch.vmap(rotate_about_axis, in_dims=(0, 0, 0))

def rotate_about_axes(vectors, axes, angles):
    '''Rotates vectors about an axis by an angle.
    Args:
        vectors (torch.Tensor of shape (n_vectors, 3)): The vectors to rotate.
        axes (torch.Tensor of shape (n_vectors, 3)): The axes of rotation: must be normalized!!
        angles (torch.Tensor of shape (n_vectors,)): The angles of rotation, using the right-hand rule.
        
    Returns:
        vectors_rotated torch.Tensor of shape (n_vectors, 3): The rotated vectors.
    '''
    vectors_rotated = vectors * torch.cos(angles).unsqueeze(1) + torch.cross(axes, vectors, dim=-1) * torch.sin(angles).unsqueeze(1) + axes * torch.sum(vectors * axes, dim=-1, keepdim=True) * (1.0 - torch.cos(angles).unsqueeze(1))
    return vectors_rotated

def reflect_about_plane(pts, plane_normal):
    plane_normal = plane_normal / torch.linalg.norm(plane_normal)
    return pts - 2 * torch.sum(pts * plane_normal, dim=1).reshape(-1, 1) * plane_normal

def rotation_matrix_from_vectors(vec1, vec2):
    """ 
    Find the rotation matrix that aligns vec1 to vec2 using PyTorch.
    Adapted from: https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space

    Args:
        vec1 (torch.Tensor): A 3D "source" vector.
        vec2 (torch.Tensor): A 3D "destination" vector.

    Returns:
        torch.Tensor: A 3x3 transformation matrix which, when applied to vec1, aligns it with vec2 — apply as torch.matmul(rotation_matrix, vec1) or torch.matmul(vec1, rotation_matrix.T).
    """
    a, b = (vec1 / torch.linalg.norm(vec1)).reshape(3), (vec2 / torch.linalg.norm(vec2)).reshape(3)
    if torch.dot(a, b) > 1 - 1e-8:
        return torch.eye(3)
    v = torch.cross(a, b)
    c = torch.dot(a, b)
    s = torch.linalg.norm(v)
    kmat = torch.tensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = torch.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
    return rotation_matrix

def project_vector_on_plane_along_direction(v, n, u):
    """
    Projects a vector v onto the plane defined by the normal n along the direction u.
    Source: https://math.stackexchange.com/questions/4108428/how-to-project-vector-onto-a-plane-but-not-along-plane-normal

    Args:
        v (torch.Tensor of shape (3,)): The vector to project.
        n (torch.Tensor of shape (3,)): The normal of the plane.
        u (torch.Tensor of shape (3,)): The direction along which to project.

    Returns:
        torch.Tensor of shape (3,): The projected vector.
    """
    # No need to normalize n and u here since the norms cancel out
    return v - ((v @ n) / (u @ n)) * u

def project_vectors_on_planes_along_directions(v, n, u):
    """
    Projects the vectors v onto the planes defined by the normals n along the directions u.
    Source: https://math.stackexchange.com/questions/4108428/how-to-project-vector-onto-a-plane-but-not-along-plane-normal

    Args:
        v (torch.Tensor of shape (..., 3)): The vectors to project.
        n (torch.Tensor of shape (..., 3)): The normals of the plane.
        u (torch.Tensor of shape (..., 3)): The directions along which to project.

    Returns:
        torch.Tensor of shape (..., 3): The projected vectors.
    """
    # No need to normalize n and u here since the norms cancel out
    return v - (torch.sum(v * n, dim=-1, keepdim=True) / torch.sum(u * n, dim=-1, keepdim=True)) * u

def symmetric_chamfer_distance_squared(vertices1, vertices2):
    '''Computes the symmetric chamfer distance between two sets of vertices.
    
    Args:
        vertices1 (torch.Tensor of shape (n_vertices1, 3)): The first set of vertices.
        vertices2 (torch.Tensor of shape (n_vertices2, 3)): The second set of vertices.
    
    Returns:
        obj (torch.Tensor of shape (,)): The symmetric chamfer distance between the two sets of vertices.
    '''
    dists_sq = torch.sum((vertices1.unsqueeze(0) - vertices2.unsqueeze(1)) ** 2, dim=-1)
    obj = torch.mean(torch.min(dists_sq, dim=1).values) + torch.mean(torch.min(dists_sq, dim=0).values)
    
    return obj

def point_mesh_squared_distance(pts, V_surf, F_surf):
    """
    Compute the squared distance between each point in pts and the mesh defined by V and F.
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/point_mesh_distance.html

    Args:
        pts (torch.Tensor of shape (N, 3)): The points.
        V_surf (torch.Tensor of shape (V, 3)): The vertices of the mesh.
        F_surf (torch.Tensor of shape (F, 3)): The faces of the mesh.
    """
    from pytorch3d.structures import Meshes, Pointclouds
    from pytorch3d.loss.point_mesh_distance import point_face_distance

    # Cast to float32: pytorch3d requires float32
    pts = pts.to(torch.float32)
    V_surf = V_surf.to(torch.float32)
    F_surf = F_surf.to(torch.int64)

    meshes = Meshes(verts=[V_surf], faces=[F_surf])
    pcls = Pointclouds(points=[pts])

    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()

    # point to face distance: shape (P,)
    point_to_face = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points
    )

    return torch.sum(point_to_face)

def sin_angle(a, v1, v2):
    """
    Get the sin of the signed angle from v1 to v2 around axis "a". Uses right hand rule
    as the sign convention: clockwise is positive when looking along vector.
    Assumes all vectors are normalized.

    Args:
        a (torch.Tensor of shape (..., 3)): The axis of rotation.
        v1 (torch.Tensor of shape (..., 3)): The first vector.
        v2 (torch.Tensor of shape (..., 3)): The second vector.

    Returns:
        torch tensor of shape (...,): The sin of the signed angle.
    """
    return torch.sum(torch.cross(v1, v2, dim=-1) * a, dim=-1)

def angle_around_axis(a, v1, v2):
    """
    Get the signed angle from v1 to v2 around axis "a". Uses right hand rule
    as the sign convention: clockwise is positive when looking along vector.
    Assumes all vectors are normalized **and perpendicular to a**.
    Return answer in the range [-pi, pi].

    Args:
        a (torch.Tensor of shape (..., 3)): The axis of rotation.
        v1 (torch.Tensor of shape (..., 3)): The first vector.
        v2 (torch.Tensor of shape (..., 3)): The second vector.

    Returns:
        torch tensor of shape (...,): The signed angle in the range [-pi, pi].
    """
    s = torch.clamp(sin_angle(a, v1, v2), -1.0, 1.0)
    c = torch.clamp(torch.sum(v1 * v2, dim=-1), -1.0, 1.0)
    return torch.atan2(s, c)

def triangle_normal(v1, v2, v3):
    '''
    Args:
        v1 (torch.Tensor of shape (..., 3)): The first vertex of the triangle.
        v2 (torch.Tensor of shape (..., 3)): The second vertex of the triangle.
        v3 (torch.Tensor of shape (..., 3)): The third vertex of the triangle.
        
    Returns:
        n (torch.Tensor of shape (..., 3)): The normal of the triangle.
    '''
    # no need for normalization here, the normal is normalized in the next step
    n = torch.cross(v2 - v1, v3 - v1, dim=-1)
    n = n / torch.linalg.norm(n, dim=-1, keepdim=True)
    return n

def triangle_area(v1, v2, v3):
    '''Compute the area of a triangle defined by the points v1, v2, and v3.

    Args:
        v1 (torch.Tensor of shape (..., 3)): The first vertex of the triangle.
        v2 (torch.Tensor of shape (..., 3)): The second vertex of the triangle.
        v3 (torch.Tensor of shape (..., 3)): The third vertex of the triangle.

    Returns:
        area (torch.Tensor of shape (...,)): The area of the triangle.
    '''
    a = torch.linalg.norm(v2 - v1, dim=-1, keepdim=True)
    b = torch.linalg.norm(v3 - v2, dim=-1, keepdim=True)
    c = torch.linalg.norm(v1 - v3, dim=-1, keepdim=True)
    s = 0.5 * (a + b + c)
    return torch.sqrt(s * (s - a) * (s - b) * (s - c))

def triangulate_polygon(pts):
    '''Triangulate a polygon defined by the points pts.
    
    Args:
        pts (torch.Tensor of shape (n, 3)): The points defining the polygon.
        
    Returns:
        faces (torch.Tensor of shape (n-2, 3)): The indices of the vertices of the triangles.
    '''
    n = pts.shape[0]
    faces = torch.zeros((n-2, 3), dtype=torch.int64)
    faces[:, 1] = torch.arange(1, n-1)
    faces[:, 2] = torch.arange(2, n)
    return faces

def compute_polygonal_area(x, y):
    '''Shoelace formula, source: https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates

    Args:
        x (torch.Tensor of shape (..., n)): The x-coordinates of the polygon vertices.
        y (torch.Tensor of shape (..., n)): The y-coordinates of the polygon vertices.

    Returns:
        area (torch.Tensor of shape (...,)): The area of the polygon(s).
    '''
    return 0.5 * torch.abs(torch.sum(x * torch.roll(y, 1, dims=-1), dim=-1) - torch.sum(y * torch.roll(x, 1, dims=-1), dim=-1))

def compute_curve_length(directrix):
    '''Compute the length of a curve defined by directrix.
    
    Args:
        directrix (torch.Tensor of shape (M, 3)): The points defining the curve.

    Returns:
        length (torch.Tensor of shape ()): The length of the curve.
    '''
    return torch.sum(torch.linalg.norm(directrix[1:] - directrix[:-1], dim=1))

def compute_min_curve_self_distance(points: torch.Tensor, neighborhood_arclength_dist: float, closed: bool = False) -> torch.Tensor:
    """
    For each point in a polyline, compute the minimum 3D distance to other non-neighboring points,
    where "neighboring" means that their distance along the curve is less than neighborhood_arclength_dist.

    Args:
        points: (n, 3) tensor of 3D points forming the polyline.
        neighborhood_arclength_dist: Scalar. Points within this arc length distance are ignored.
        closed: Bool. If True, treat the polyline as a closed curve.
    
    Returns:
        min_dists: (n,) tensor of minimum 3D distances to non-neighboring points.
    """
    n = points.shape[0]

    seg_lens = torch.norm(points[1:] - points[:-1], dim=1)  # (n-1,)

    # Compute cumulative arc lengths
    arc_lengths = torch.zeros(n, dtype=points.dtype, device=points.device)
    arc_lengths[1:] = torch.cumsum(seg_lens[:n-1], dim=0) if not closed else torch.cumsum(seg_lens, dim=0)

    # Total curve length
    total_length = arc_lengths[-1] + (seg_lens[-1] if not closed else 0)

    # Compute pairwise 3D distances
    diff = points.unsqueeze(1) - points.unsqueeze(0)  # (n, n, 3)
    dist_matrix = torch.norm(diff, dim=2)  # (n, n)

    # Compute pairwise arc distances (cyclic if closed)
    arc_i = arc_lengths.view(-1, 1)  # (n, 1)
    arc_j = arc_lengths.view(1, -1)  # (1, n)
    arc_diff = torch.abs(arc_i - arc_j)

    if closed:
        arc_diff = torch.minimum(arc_diff, total_length - arc_diff)  # cyclic distance

    # Mask out neighboring points (i.e., arc distance too small or diagonal)
    mask = arc_diff >= neighborhood_arclength_dist
    mask.fill_diagonal_(False)

    # Apply mask
    dist_matrix_masked = torch.where(mask, dist_matrix, torch.full_like(dist_matrix, float('inf')))

    # Compute minimum valid 3D distance per point
    min_dists, _ = torch.min(dist_matrix_masked, dim=1)

    return min_dists

def compute_angle_mismatch(t1, t2):
    '''Compute the mismatch in curvature angles between two polygons. 
    Similar polygons result in zero mismatch irrespective of their scaling factor.
    
    Args:
        t1 (torch.Tensor of shape (n, 3)): The first polygon.
        t2 (torch.Tensor of shape (n, 3)): The second polygon.

    Returns:
        mismatch (torch.Tensor of shape ()): The mismatch in curvature angles.
    '''

    assert t1.shape[1] == t2.shape[1] == 3
    first_last_overlap_t1 = torch.le(torch.linalg.norm(t1[0] - t1[-1]), torch.tensor(1.0e-5))
    first_last_overlap_t2 = torch.le(torch.linalg.norm(t2[0] - t2[-1]), torch.tensor(1.0e-5))
    
    curv_angles_shape = compute_curvature_angles(t1, closed_curve=True, first_last_overlap=first_last_overlap_t1)
    curv_angles_target = compute_curvature_angles(t2, closed_curve=True, first_last_overlap=first_last_overlap_t2)
    angles_shape = torch.pi - curv_angles_shape
    angles_target = torch.pi - curv_angles_target
    n_angles = angles_shape.shape[0]
    idx_circ = (n_angles - torch.arange(n_angles)[None].T + torch.arange(n_angles)[None]) % n_angles # to be used to create a circulant matrix of angles_shape
    mismatch = torch.min(torch.sum((angles_shape[idx_circ] - angles_target.reshape(1, -1)) ** 2, dim=1))
    return mismatch

def is_quad_planar(vertices):
    '''Check if a quad is planar.
    
    Args:
        vertices (torch.Tensor of shape (4, 3)): The vertices of the quad.

    Returns:
        bool: True if the quad is planar, False otherwise.
    '''
    v1 = vertices[1] - vertices[0]
    v2 = vertices[2] - vertices[0]
    v3 = vertices[3] - vertices[0]
    na = torch.cross(v1, v2, dim=-1)
    na = na / torch.linalg.norm(na)
    nb = torch.cross(v2, v3, dim=-1)
    nb = nb / torch.linalg.norm(nb)
    atol = 1.0e-12
    if torch.allclose(na, nb, atol=atol):
        return True
    elif torch.allclose(na, -nb, atol=atol):
        print("WARNING: The computed normals of a quad have opposite signs, there might have been an inversion in the quad.")
        return True  # the quad can still be considered planar
    return False

def align_point_cloud(points, pure_rotation=True):
    '''Aligns point clouds to xyz axes'''
    centered_points = points - torch.mean(points, dim=0, keepdim=True)
    cov = torch.einsum("si, sj -> ij", centered_points, centered_points)
    L, Q = torch.linalg.eigh(cov)
    Q = Q[:, torch.argsort(L, descending=True)]

    if pure_rotation and torch.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]

    rotated_pos = centered_points @ Q
    return rotated_pos
    
def point_cloud_is_planar(pts, tol=1e-8):
    '''Check if the point cloud is planar.'''
    if pts.shape[0] < 3:  # The point cloud has less than 3 points
        return True
    if pts.shape[1] == 2:  # The point cloud is 2D
        return True
    pts_centered = pts - torch.mean(pts, dim=0)
    cov = pts_centered.T @ pts_centered
    eigvals, eigvecs = torch.linalg.eig(cov)
    return torch.min(torch.abs(eigvals)) < tol

def find_shared_points(pts1, pts2, tol=1e-5):
    """
    Find the indices of the points that are present in both the given sets within a certain tolerance.

    Args:
        pts1 (torch.Tensor (n, 3)): The first set of 3D points.
        pts2 (torch.Tensor (m, 3)): The second set of 3D points.
        tol (float): The tolerance within which two points are considered to be the same.

    Returns:
        shared_indices (List[List[int]]): The indices of the shared points in the format [[i1, j1], [i2, j2], ...].
    """
    diff = pts1[:, None, :] - pts2[None, :, :]
    dist = torch.linalg.norm(diff, dim=2)
    shared_indices = torch.nonzero(dist < tol, as_tuple=False)
    return shared_indices.tolist()

def find_duplicate_points(pts, tol=1e-5):
    """
    Find the indices of the duplicate points within a certain tolerance.

    Args:
        pts (torch.Tensor (n, 3)): The set of 3D points.
        tol (float): The tolerance within which two points are considered to be the same.

    Returns:
        duplicate_indices (List[List[int]]): The indices of the duplicate points in the format [[i1, i2], [i3, i4], ...].
    """
    diff = pts[:, None, :] - pts[None, :, :]
    dist = torch.linalg.norm(diff, dim=2)
    duplicate_indices = torch.nonzero(dist < tol, as_tuple=False)
    duplicate_indices = duplicate_indices[duplicate_indices[:, 0] != duplicate_indices[:, 1]]
    duplicate_indices = duplicate_indices[duplicate_indices[:, 0] < duplicate_indices[:, 1]]  # remove flipped pairs: only keep one of the two
    return duplicate_indices.tolist()

# --------------------------------------------------------------------------------
# Curve utils
# --------------------------------------------------------------------------------

def compute_tangents(directrix):
    '''Computes the edge tangents of a curve defined by directrix.

    Args:
        directrix (torch.Tensor of shape (M, 3)): The points defining the curve.

    Returns:
        tangents (torch.Tensor of shape (M-1, 3)): The tangent vectors on the edges.
    '''
    tangents = directrix[1:] - directrix[:-1]
    tangents = tangents / torch.linalg.norm(tangents, dim=1, keepdim=True)
    return tangents

def compute_plane_normals(directrix, kind='bisecting', closed_curve=False):
    '''Computes the vertex tangents of a curve defined by directrix.

    Args:
        directrix (torch.Tensor of shape (M, 3)): 
            The points defining the curve.
        kind (str, optional): 
            If 'bisecting', the vertex tangent is computed by bisectrix of the curvature angle. 
            If 'integrated', the vertex tangent is computed by integrating the tangent along the voronoi cell.
            If 'reflective', the vertex tangent is the normal of the plane that reflects one edge tangent to the next. 
                It is computed by rotating the bisecting tangent by 90˚ about the binormal 
        closed_curve (bool, optional): 
            If True, the curve is assumed to be closed. The first and last points must be the same.

    Returns:
        vertex_tangents (torch.Tensor of shape (M, 3)): The tangent vectors at the vertices.
    '''
    tangents = compute_tangents(directrix)
    if kind == 'integrated':
        vertex_tangents = torch.cat([(directrix[1] - directrix[0]).unsqueeze(0), directrix[2:] - directrix[:-2], (directrix[-1] - directrix[-2]).unsqueeze(0)], dim=0)
        vertex_tangents = vertex_tangents / torch.linalg.norm(vertex_tangents, dim=1, keepdim=True)
    elif kind == 'bisecting' or kind == 'reflective':
        if closed_curve:
            vt0 = (tangents[0] + tangents[-1]).unsqueeze(0)
            vtnm1 = vt0
        else:   # use edge tangent for the first and last vertex
            vt0 = tangents[0].unsqueeze(0)
            vtnm1 = tangents[-1].unsqueeze(0)
        if kind == 'bisecting':
            vertex_tangents = torch.cat([vt0, tangents[1:] + tangents[:-1], vtnm1], dim=0)
        elif kind == 'reflective':
            vertex_tangents = torch.cat([vt0, tangents[1:] - tangents[:-1], vtnm1], dim=0)
        vertex_tangents = vertex_tangents / torch.linalg.norm(vertex_tangents, dim=1, keepdim=True)
    else:
        raise ValueError("kind must be 'integrated', 'bisecting', or 'reflective'.")
    return vertex_tangents

def compute_binormals(directrix, closed_curve=False):
    '''Computes the binormals of a curve defined by directrix.
    
    Args:
        directrix (torch.Tensor of shape (M, 3)): The points defining the curve.
        closed_curve (bool): If True, the curve is assumed to be closed. The first and last points must be the same.

    Returns:
        binormals (torch.Tensor of shape (M, 3)): The binormal vectors at the vertices.
    '''
    tangents = compute_tangents(directrix)
    if closed_curve:
        b0 = torch.cross(tangents[-1], tangents[0], dim=0)
        bnm1 = b0
    else:  # assign the edge binormal of the second vertex to the first one; same at the end
        b0 = torch.cross(tangents[0], tangents[1], dim=0)
        bnm1 = torch.cross(tangents[-2], tangents[-1], dim=0)
    binormals = torch.cat([b0.unsqueeze(0), torch.cross(tangents[:-1], tangents[1:], dim=1), bnm1.unsqueeze(0)], dim=0)
    binormals = binormals / torch.linalg.norm(binormals, dim=1, keepdim=True)
    return binormals

def compute_curvature_angles(directrix, closed_curve=False, first_last_overlap=False):
    '''Computes the curvature angles of a curve defined by directrix.

    Args:
        directrix (torch.Tensor of shape (M, 3)): The points defining the curve.
        closed_curve (bool): If True, the curve is assumed to be closed. The first and last points need not be the same.

    Returns:
        curvature_angles (torch.Tensor of shape (M-closed_curve,)): The curvature angles at the vertices. It has zeros at the first and last points if closed_curve is False.
    '''
    assert not first_last_overlap or torch.le(torch.linalg.norm(directrix[0] - directrix[-1]), torch.tensor(1.0e-5)), "First and last points must be the same for closed curves with overlapping ends."
    if closed_curve and not first_last_overlap:
        tangents_non_norm = (torch.roll(directrix, -1, dims=0) - directrix)
        tangents = tangents_non_norm / torch.linalg.norm(tangents_non_norm, dim=1, keepdim=True)
    else:
        tangents = (directrix[1:] - directrix[:-1]) / torch.linalg.norm(directrix[1:] - directrix[:-1], dim=1, keepdim=True)

    if closed_curve:
        normal = torch.cross(tangents, torch.roll(tangents, -1, dims=0), dim=1)
        curvature_angles = angle_around_axis(normal, tangents, torch.roll(tangents, -1, dims=0))
    else:
        normal = torch.cross(tangents[:-1], tangents[1:], dim=1)
        curvature_angles = torch.zeros_like(directrix[:, 0]) # so that curvature_angles is a batched tensor when vmap is used
        curvature_angles[1:-1] = angle_around_axis(normal, tangents[:-1], tangents[1:])
    return curvature_angles
vmap_compute_curvature_angles = torch.vmap(compute_curvature_angles, in_dims=(0, None, None))

def compute_torsion_angles(directrix, closed_curve=False):
    '''Computes the torsion angles of a curve defined by directrix.

    Args:
        directrix (torch.Tensor of shape (M, 3)): The points defining the curve.
        closed_curve (bool): If True, the curve is assumed to be closed. The first and last points need not be the same.

    Returns:
        torsion_angles (torch.Tensor of shape (M-1,)): The torsion angles at the edges. It has zeros at the first and last edges if closed_curve is False.
    '''
    M = directrix.shape[0]
    first_last_overlap = torch.le(torch.linalg.norm(directrix[0] - directrix[-1]), torch.tensor(1.0e-5))
    if closed_curve and not first_last_overlap:
        tangents_non_norm = (torch.roll(directrix, -1, dims=0) - directrix)
        tangents = tangents_non_norm / torch.linalg.norm(tangents_non_norm, dim=1, keepdim=True)
    else:
        tangents = (directrix[1:] - directrix[:-1]) / torch.linalg.norm(directrix[1:] - directrix[:-1], dim=1, keepdim=True)

    if closed_curve:
        binormals = torch.cross(torch.roll(tangents, 1, dims=0), tangents, dim=1)
        torsion_angles = angle_around_axis(tangents, binormals, torch.roll(binormals, -1, dims=0))
    else:
        binormals = torch.cross(tangents[:-1], tangents[1:], dim=1)
        torsion_angles = torch.zeros(size=(M-1,))
        torsion_angles[1:-1] = angle_around_axis(tangents[1:-1], binormals[:-1], binormals[1:])
    return torsion_angles

def rotate_and_translate_profile_curve(profile_curve, translation, plane_normal, rotation_angle=0.0):
    """
    Rotate a profile curve to lie in the plane defined by the plane normal and translate it by the translation vector.
    Optionally, rotate the profile curve by the rotation angle about the plane normal.

    Args:
        profile_curve (torch.Tensor): The profile curve to rotate and translate. Shape (n_points, 2) or (n_points, 3).
        translation (torch.Tensor): The translation vector. Shape (3,).
        plane_normal (torch.Tensor): The normal vector of the plane. Shape (3,).
        rotation_angle (float): The angle to rotate the profile curve about the plane normal. Default is 0.0.

    Returns:
        torch.Tensor: The rotated and translated profile curve. Shape (n_points, 3).
    """
    assert profile_curve.shape[1] == 2 or (profile_curve.shape[1] == 3 and torch.allclose(profile_curve[:, 2], 0.0)), "Profile curve must be 2D or 3D with z-coordinates equal to zero."
    if profile_curve.shape[1] == 2:
        profile_curve = torch.cat([profile_curve, torch.zeros_like(profile_curve[:, 0]).unsqueeze(1)], dim=1)

    if rotation_angle != 0.0:
        profile_curve = rotate_about_axis(profile_curve, plane_normal, rotation_angle)
    
    plane_normal = plane_normal / torch.linalg.norm(plane_normal)
    e1 = torch.cross(plane_normal, torch.tensor([0.0, 0.0, 1.0]), dim=0)
    e1 = e1 / torch.linalg.norm(e1)
    e2 = torch.cross(plane_normal, e1, dim=0)
    rot_xyz_to_frame = torch.zeros(size=(3, 3))
    rot_xyz_to_frame[:, 0] = e1
    rot_xyz_to_frame[:, 1] = e2
    rot_xyz_to_frame[:, 2] = plane_normal
    return translation + profile_curve @ rot_xyz_to_frame.T

def resample_polyline(pts, n, sample_map=None):
    """
    Resample a polyline defined by the points pts to have n points approximately equispaced along the curve.
    sample_map is a function that maps the cumulative distance to the new distance. If None, the points are spaced equally.
    """
    from scipy import interpolate
    import numpy as np

    # Convert pts to numpy array
    pts_np = pts.numpy()

    # Compute the cumulative arc length along the polyline
    distances = np.sqrt(np.sum(np.diff(pts_np, axis=0)**2, axis=1))
    cumulative_distances = np.concatenate(([0], np.cumsum(distances)))

    # Create an interpolation function for each dimension
    interpolators = [interpolate.interp1d(cumulative_distances, pts_np[:, dim]) for dim in range(pts_np.shape[1])]

    # Generate n equally spaced points along the cumulative distance
    new_distances = np.linspace(0, cumulative_distances[-1], n)
    if sample_map is not None:
        new_distances = sample_map(new_distances)

    # Interpolate the points at the new distances
    resampled_pts_np = np.vstack([interpolator(new_distances) for interpolator in interpolators]).T

    # Convert resampled points back to torch tensor
    resampled_pts = torch.tensor(resampled_pts_np, dtype=pts.dtype)

    return resampled_pts

def regular_polygon(n_sides, radius=1.0):
    '''Generate a regular polygon with n_sides and given radius.

    Args:
        n_sides (int): The number of sides of the polygon.
        radius (float): The radius of the circumcircle of the polygon.

    Returns:
        torch.Tensor of shape (n_sides, 2): The vertices of the polygon in the xy-plane.
    '''
    import numpy as np

    t = torch.linspace(0.0, 2.0 * np.pi, n_sides+1)[:-1]
    x = radius * torch.cos(t)
    y = radius * torch.sin(t)
    return torch.stack([x, y], dim=1)

# --------------------------------------------------------------------------------
# Tube utils
# --------------------------------------------------------------------------------

def is_tube_pq(ctube_vertices):
    '''Check if the tube is a PQ-tube, i.e. each of its quad faces is planar.

    Args:
        ctube_vertices (torch.Tensor of shape (M, N, 3)): The vertices of the tube.   
    '''
    from Ctubes.tube_generation import compute_ctube_topology

    M = ctube_vertices.shape[0]
    N = ctube_vertices.shape[1]
    quads, _ = compute_ctube_topology(M, N)
    ctube_vertices_reshaped = ctube_vertices.reshape(-1, 3)
    n_quads = quads.shape[0]  # 3*(M-1)
    for i in range(n_quads):
        quad = ctube_vertices_reshaped[quads[i]]
        if not is_quad_planar(quad):
            return False
    return True

def compute_quad_nonplanarity(vertices):
    '''
    Compute nonplanarity according to the measure from [Pottmann et al. 2008].
    Given the 4 vertices of a quad as q1, q2, p1, p2, the nonplanarity is defined as the distance between the two lines:
        l1: p1 v q2
    and 
        l2: q1 v p2
    where (a v b) denotes the line spanned by the two points a and b.

    Args:
        vertices: torch.Tensor of shape (M, N, 3)

    Returns:
        torch.Tensor of shape (M-1, N)
    '''

    # Helper functions
    def line_coefficients(point1: torch.Tensor, point2: torch.Tensor):
        """
        Compute the coefficients of the line passing through two points in 3D space.

        Args:
            point1 (torch.Tensor): A tensor of shape (3,) representing the first point (x1, y1, z1).
            point2 (torch.Tensor): A tensor of shape (3,) representing the second point (x2, y2, z2).

        Returns:
            r0 (torch.Tensor): A tensor representing the starting point (point1).
            v (torch.Tensor): A tensor representing the direction vector of the line.
        """
        if point1.shape != (3,) or point2.shape != (3,):
            raise ValueError("Both points must be 3D tensors with shape (3,).")

        # r0 is simply the first point
        r0 = point1

        # v is the direction vector from point1 to point2
        v = point2 - point1

        return r0, v

    def minimal_distance_between_lines(r0_1: torch.Tensor, v1: torch.Tensor, r0_2: torch.Tensor, v2: torch.Tensor):
        """
        Compute the minimal distance between two lines in 3D space.

        Args:
            r0_1 (torch.Tensor): A tensor representing a point on the first line.
            v1 (torch.Tensor): A tensor representing the direction vector of the first line.
            r0_2 (torch.Tensor): A tensor representing a point on the second line.
            v2 (torch.Tensor): A tensor representing the direction vector of the second line.

        Returns:
            float: The minimal distance between the two lines.
        """
        if r0_1.shape != (3,) or v1.shape != (3,) or r0_2.shape != (3,) or v2.shape != (3,):
            raise ValueError("All inputs must be 3D tensors with shape (3,).")

        # Compute the cross product of the direction vectors
        cross_v1_v2 = torch.cross(v1, v2, dim=0)

        # If the cross product is zero, the lines are parallel or coincident
        if torch.linalg.norm(cross_v1_v2) == 0:
            # Compute the distance between a point on one line to the other line
            # Since the lines are parallel, pick any point on the second line
            diff = r0_2 - r0_1
            projection = torch.dot(diff, v1) / torch.linalg.norm(v1)
            closest_point_on_line1 = r0_1 + projection * (v1 / torch.linalg.norm(v1))
            return torch.linalg.norm(closest_point_on_line1 - r0_2).item()

        # Compute the vector between points on the two lines
        diff_r0 = r0_2 - r0_1

        # Minimal distance is the projection of diff_r0 onto the unit normal vector of the plane formed by v1 and v2
        distance = torch.abs(torch.dot(diff_r0, cross_v1_v2)) / torch.linalg.norm(cross_v1_v2)

        return distance.item()

    # Compute the nonplanarity
    M, N, _ = vertices.shape
    nonplanarities = []
    for i in range(1, M):
        for j in range(N):
            p1 = vertices[i-1, j]
            q1 = vertices[i-1, (j+1) % N]
            p2 = vertices[i, j]
            q2 = vertices[i, (j+1) % N]
            r0_1, v1 = line_coefficients(p1, q2)
            r0_2, v2 = line_coefficients(q1, p2)
            nonplanarities.append(minimal_distance_between_lines(r0_1, v1, r0_2, v2))
    return torch.tensor(nonplanarities).reshape(M-1, N)

def compute_cross_section_areas(ctube_vertices):
    '''Compute the areas of the swept cross-sections.'''
    M = ctube_vertices.shape[0]
    ctube_vertices_centered = ctube_vertices - torch.mean(ctube_vertices, dim=1, keepdim=True)
    normals = torch.cross(ctube_vertices_centered[:, 0, :], ctube_vertices_centered[:, 1, :], dim=1) # the triangle is centered
    normals = normals / torch.linalg.norm(normals, dim=1, keepdim=True)
    # Define the rotation that aligns the normal to the z-axis
    e1 = ctube_vertices_centered[:, 0, :] / torch.linalg.norm(ctube_vertices_centered[:, 0, :], dim=1, keepdim=True)
    rotations = torch.zeros(size=(M, 3, 3))
    rotations[:, :, 0] = e1
    rotations[:, :, 1] = torch.cross(normals, e1, dim=1)
    rotations[:, :, 2] = normals
    ctube_vertices_centered_xy_plane = torch.einsum("ilj,ikl->ikj", rotations, ctube_vertices_centered)
    areas = compute_polygonal_area(ctube_vertices_centered_xy_plane[:, :, 0], ctube_vertices_centered_xy_plane[:, :, 1])
    return areas

def compute_cross_section_radii(directrix, ctube_vertices, on_normal_plane=True):
    '''Compute the radii of the swept cross-sections as half of the maximum distance between any two vertices of the cross-section.

    Args:
        directrix (torch.Tensor of shape (M, 3)): The points defining the curve.
        ctube_vertices (torch.Tensor of shape (M, N, 3)): The vertices of the tube.
        on_normal_plane (bool): If True, the cross-sections are first projected onto the normal bisecting planes before computing the distances.

    Returns:
        torch.Tensor of shape (M,): The radii of the cross-sections.
    '''
    M = ctube_vertices.shape[0]
    if on_normal_plane:
        # Assume that the curve is closed if 1) the first and the last points are the same AND 2) the first and the last cross-sections have the same normal
        n0 = triangle_normal(*ctube_vertices[0, 0:3])
        n1 = triangle_normal(*ctube_vertices[-1, 0:3])
        closed_curve = torch.allclose(directrix[0], directrix[-1]) and (torch.allclose(n0, n1) or torch.allclose(n0, -n1))

        # Project the vertices onto the normal bisecting planes before computing the distances
        vertex_tangents = compute_plane_normals(directrix, kind='bisecting', closed_curve=closed_curve)
        ctube_vertices_on_normal_plane = directrix.reshape(M, 1, 3) + project_vectors_on_planes_along_directions(ctube_vertices - directrix.reshape(M, 1, 3), vertex_tangents.reshape(M, 1, 3), vertex_tangents.reshape(M, 1, 3))
        dist_matrix = torch.cdist(ctube_vertices_on_normal_plane, ctube_vertices_on_normal_plane) # (M, N, N)
    else:
        dist_matrix = torch.cdist(ctube_vertices, ctube_vertices) # (M, N, N)
    
    radii = 0.5 * smooth_max(dist_matrix.reshape(M, -1), dim=1, p=50) # (M,)

    return radii

def compute_quad_normals(vertices, quad_tube_indices, quad_disc_indices, quad_cross_section_indices):
    '''Compute the normal of a (list of) quads.

    Args:
        vertices (list of torch Tensors of shape (M, N, 3)): The vertices of the tubes, one tensor per tube.
        quad_tube_indices (torch.Tensor of shape (n_quads,)): The indices of the tubes.
        quad_disc_indices (torch.Tensor of shape (n_quads,)): The indices of the directrix points.
        quad_cross_section_indices (torch.Tensor of shape (n_quads,)): The indices of the cross-sections.
    '''
    assert quad_tube_indices.shape == quad_disc_indices.shape == quad_cross_section_indices.shape, "The indices must have the same shape."
    n_quads = quad_tube_indices.shape[0]
    vertices_quads = torch.zeros(size=(n_quads, 3, 3)) # disregard the last vertex
    for i in range(n_quads):
        ti = quad_tube_indices[i]
        di = quad_disc_indices[i]
        ci = quad_cross_section_indices[i]
        ci1 = (ci + 1) % vertices[ti].shape[1]  # Wrap around (assumes closed cross-sections)
        vertices_quads[i, 0] = vertices[ti][di, ci]
        vertices_quads[i, 1] = vertices[ti][di, ci1]
        vertices_quads[i, 2] = vertices[ti][di+1, ci1]
    normals = torch.cross(vertices_quads[:, 1] - vertices_quads[:, 0], vertices_quads[:, 2] - vertices_quads[:, 0], dim=1)
    return normals / (1.0e-12 + torch.linalg.norm(normals, dim=1, keepdim=True))

def compute_quad_centers(vertices, quad_tube_indices, quad_disc_indices, quad_cross_section_indices):
    '''Compute the center of a (list of) quads.

    Args:
        vertices (list of torch Tensors of shape (M, N, 3)): The vertices of the tubes, one tensor per tube.
        quad_tube_indices (torch.Tensor of shape (n_quads,)): The indices of the tubes.
        quad_disc_indices (torch.Tensor of shape (n_quads,)): The indices of the directrix points.
        quad_cross_section_indices (torch.Tensor of shape (n_quads,)): The indices of the cross-sections.

    Returns:
        torch.Tensor of shape (n_quads, 3): The centers of the quads.
    '''
    assert quad_tube_indices.shape == quad_disc_indices.shape == quad_cross_section_indices.shape, "The indices must have the same shape."
    n_quads = quad_tube_indices.shape[0]

    vertices_quads = torch.zeros(size=(n_quads, 4, 3))
    for i in range(n_quads):
        ti = quad_tube_indices[i]
        di = quad_disc_indices[i]
        ci = quad_cross_section_indices[i]
        ci1 = (ci + 1) % vertices[ti].shape[1]  # Wrap around (assumes closed cross-sections)
        vertices_quads[i, 0] = vertices[ti][di, ci]
        vertices_quads[i, 1] = vertices[ti][di, ci1]
        vertices_quads[i, 2] = vertices[ti][di+1, ci1]
        vertices_quads[i, 3] = vertices[ti][di+1, ci]
    return torch.mean(vertices_quads, dim=1)

def get_bisecting_plane_normals_with_symmetry(directrix):
    assert directrix.has_symmetry()
    symmetry_transforms = directrix.symmetry_transforms
    transform_first = symmetry_transforms[0]
    transform_last = symmetry_transforms[-1]

    X_a = directrix.X
    X_b = transform_first(directrix.X)
    X_last = transform_last(directrix.X)
    assert torch.allclose(X_last[-1], X_a[0]), "Currently assuming a closed symmetric directrix."
    assert torch.allclose(X_b[0], X_a[-1]), "The symmetrized curves do not match"

    pn = compute_plane_normals(directrix.X, kind='bisecting', closed_curve=False)

    # Symmetrize the first and last plane normals
    pn_last = ((X_a[-1] - X_a[-2]) + (X_b[1] - X_b[0])).reshape(1, 3)
    pn_last = pn_last / torch.linalg.norm(pn_last)
    pn_first = transform_last(pn_last)
    pn[-1] = pn_last
    pn[0] = pn_first

    return pn

# --------------------------------------------------------------------------------
# Parallel transport
# --------------------------------------------------------------------------------

def parallel_transport_normalized(t0, t1, v):
    '''Parallel transports a vector v from the tangent t0 to the tangent t1. Assumes t0 and t1 are normalized.

    Note: This is probably slower when t0 and t1 align since res is always evaluated, but it is assumed not to happen often in the context of this project.
    
    Args:
        t0: torch.tensor of shape (..., 3), the *normalized* tangent vector at the starting point.
        t1: torch.tensor of shape (..., 3), the *normalized* tangent vector at the ending point.
        v: torch.tensor of shape (..., 3), the vector to be parallel transported.

    Returns:
        torch.tensor of shape (..., 3), the parallel transported vector.
    '''
    sin_theta_axis = torch.cross(t0, t1, dim=-1)
    cos_theta = torch.sum(t0 * t1, dim=-1, keepdim=True)
    den = 1.0 + cos_theta
    mask_identity = torch.logical_or(torch.abs(den) < 1.0e-14, torch.linalg.norm(t0 - t1, dim=-1, keepdim=True) < 1.0e-6)
    sin_theta_axis_dot_v = torch.sum(sin_theta_axis * v, dim=-1, keepdim=True)
    sin_theta_axis_cross_v = torch.cross(sin_theta_axis, v, dim=-1)
    res = (sin_theta_axis_dot_v / (den + mask_identity * 1.0e-6)) * sin_theta_axis + sin_theta_axis_cross_v + cos_theta * v # Not to divide by zero when den is zero
    return mask_identity * v + (~mask_identity) * res

def parallel_transport(t0, t1, v):
    '''Parallel transports a vector v from the tangent t0 to the tangent t1.
    
    Args:
        t0: torch.tensor of shape (..., 3), the tangent vector at the starting point.
        t1: torch.tensor of shape (..., 3), the tangent vector at the ending point.
        v: torch.tensor of shape (..., 3), the vector to be parallel transported.

    Returns:
        torch.tensor of shape (..., 3), the parallel transported vector.
    '''
    t0_normalized = t0 / torch.linalg.norm(t0, dim=-1, keepdim=True)
    t1_normalized = t1 / torch.linalg.norm(t1, dim=-1, keepdim=True)
    return parallel_transport_normalized(t0_normalized, t1_normalized, v)

# --------------------------------------------------------------------------------
# Interpolations
# --------------------------------------------------------------------------------

def slerp(p0, p1, n_inter):
    '''
    Args:
        p0: torch.tensor of shape (dim,)
        p1: torch.tensor of shape (dim,)
        n_inter: int, number of output points (must be greater than 2)
        
    Returns:
        torch.tensor of shape (n_inter, dim)
    '''
    ts = torch.linspace(0.0, 1.0, n_inter).unsqueeze(1)
    omega = torch.acos(torch.dot(p0, p1) / (torch.linalg.norm(p0) * torch.linalg.norm(p1)))
    sin_omega = torch.sin(omega)
    return (torch.sin((1.0 - ts) * omega) * p0.unsqueeze(0) + torch.sin(ts * omega) * p1.unsqueeze(0)) / sin_omega

# --------------------------------------------------------------------------------
# Rotation utils
# --------------------------------------------------------------------------------

def quaternion_to_matrix(quaternions):
    '''
    Convert rotations given as quaternions to rotation matrices.
    
    Source: https://pytorch3d.readthedocs.io/en/latest/modules/transforms.html#pytorch3d.transforms.quaternion_to_matrix

    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    '''
    i, j, k, r = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def sqrt_positive_part(x):
    '''
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#_sqrt_positive_part
    '''
    ret = torch.zeros_like(x)
    positive_mask = x > 0.0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret

def standardize_quaternion(quaternions):
    '''
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.
    
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#standardize_quaternion

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    '''
    return torch.where(quaternions[..., 0:1] < 0.0, -quaternions, quaternions)

def matrix_to_quaternion(matrix):
    '''
    Convert rotations given as rotation matrices to quaternions.
    
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
    
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        
    Returns:
        quaternions with real part last, as tensor of shape (..., 4).
    '''
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)

def quaternion_to_axis_angle(quaternions):
    '''
    Convert rotations given as quaternions to axis/angle.
    
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#quaternion_to_axis_angle

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    '''
    norms = torch.linalg.norm(quaternions[..., 1:], dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2.0 * half_angles
    eps = 1.0e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48.0
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles

def axis_angle_to_quaternion(axis_angle):
    '''
    Convert rotations given as axis/angle to quaternions.
    
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#axis_angle_to_quaternion

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    '''
    angles = torch.linalg.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1.0e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48.0
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def axis_angle_to_matrix(axis_angle):
    """
    Convert rotations given as axis/angle to rotation matrices.
    
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#axis_angle_to_matrix

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def matrix_to_axis_angle(matrix):
    """
    Convert rotations given as rotation matrices to axis/angle.
    
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_axis_angle

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))

# --------------------------------------------------------------------------------
# Test
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    a = torch.tensor([torch.cos(torch.tensor(0.1)), torch.sin(torch.tensor(0.1)), 0.0]).reshape(1, 3) 
    b = torch.tensor([1.0, 0.0, 0.0]).reshape(1, 3)
    axis_rot = torch.cross(a[0], b[0], dim=0)
    norm_axis = torch.linalg.norm(axis_rot)
    cos_theta = a[0] @ b[0]
    theta = torch.atan2(norm_axis, cos_theta)
    a_rotated = rotate_about_axis(a, axis_rot/norm_axis, theta)
    print(a_rotated)