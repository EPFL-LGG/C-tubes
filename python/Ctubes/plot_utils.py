from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import os
from PIL import Image
from scipy.spatial import ConvexHull
import torch
from Ctubes.tube_generation import compute_unrolled_strips

LINEWIDTH_REF = 1.0

# --------------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------------

def get_bounding_box(points):
    '''Compute the axis-aligned bounding box of the given points.'''
    aabb_min = torch.min(points.reshape(-1, 3), dim=0).values
    aabb_max = torch.max(points.reshape(-1, 3), dim=0).values
    return aabb_min, aabb_max

def get_plot_limits(points, expand=0.3):
    '''Compute plot limits from bounding box, optionally expanding by a given fraction.'''
    aabb_min, aabb_max = get_bounding_box(points)
    extent_aabb = aabb_max - aabb_min
    extent_axes = torch.max(extent_aabb)
    aabb_center = 0.5 * (aabb_min + aabb_max)
    xlim = [aabb_center[0] - (0.5 + expand) * extent_axes, aabb_center[0] + (0.5 + expand) * extent_axes]
    ylim = [aabb_center[1] - (0.5 + expand) * extent_axes, aabb_center[1] + (0.5 + expand) * extent_axes]
    zlim = [aabb_center[2] - (0.5 + expand) * extent_axes, aabb_center[2] + (0.5 + expand) * extent_axes]
    # Convert to numpy
    xlim = [xlim[0].detach().numpy(), xlim[1].detach().numpy()]
    ylim = [ylim[0].detach().numpy(), ylim[1].detach().numpy()]
    zlim = [zlim[0].detach().numpy(), zlim[1].detach().numpy()]
    return xlim, ylim, zlim

def get_bounding_box_diagonal(points):
    aabb_min, aabb_max = get_bounding_box(points)
    aabb_diag_length = torch.linalg.norm(aabb_max - aabb_min)
    return aabb_diag_length

def extract_edges_from_faces(faces):
    """Given a list of faces, return two lists of edges, one for the boundary edges and one for the interior edges.
    The input faces can either be triangles or quads.

    Args:
        faces (torch.Tensor of shape (n_faces, n_vertices_per_face)): The faces of the mesh.

    Returns:
        boundary_edges (torch.Tensor of shape (n_boundary_edges, 2)): The boundary edges of the mesh.
        interior_edges (torch.Tensor of shape (n_interior_edges, 2)): The interior edges of the mesh.
    """

    def count_row_occurrences_in_2d_tensor(tensor, row):
        return (tensor == row).all(dim=1).sum()

    edges = []
    for face in faces:
        edges.append([face[0], face[1]])
        edges.append([face[1], face[2]])
        if len(face) == 3:  # triangle
            edges.append([face[2], face[0]])
        elif len(face) == 4:  # quad
            edges.append([face[2], face[3]])
            edges.append([face[3], face[0]])
        else:
            raise ValueError("The faces must be either triangles or quads.")
    edges = torch.tensor(edges)
    boundary_edges = []
    interior_edges = []
    for edge in edges:
        if count_row_occurrences_in_2d_tensor(edges, edge) + count_row_occurrences_in_2d_tensor(edges, edge.flip(0)) == 1:
            boundary_edges.append(edge)
        else:
            interior_edges.append(edge)
    boundary_edges = torch.cat(boundary_edges).reshape(-1, 2)
    if len(interior_edges) > 0:
        interior_edges = torch.cat(interior_edges).reshape(-1, 2)
    else:
        interior_edges = torch.empty((0, 2), dtype=torch.long)
    # NOTE: each interior edge appears twice in the list, once in each direction.
    return boundary_edges, interior_edges

def rgb_to_hex(rgb):
    '''Convert an RGB tuple to a hex string.'''
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

# --------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------

def plot_tube(directrix, ctube_vertices, fig=None, ax=None, save_path=None, xlim=None, ylim=None, target_cross_sections=None, curve_color='C0', tube_color='k', target_cross_section_color='r'):
    if fig is None and ax is None:
        fig = plt.figure(figsize=(8, 8), dpi=100)
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0, 0])
    else:
        assert fig is not None and ax is not None, "If fig and ax are provided, both must be provided."

    # No gradients needed here: detach and convert to numpy
    if isinstance(directrix, torch.Tensor):
        directrix = directrix.detach().numpy()
    if isinstance(ctube_vertices, torch.Tensor):
        ctube_vertices = ctube_vertices.detach().numpy()
    if target_cross_sections is not None and isinstance(target_cross_sections, torch.Tensor):
        target_cross_sections = target_cross_sections.detach().numpy()

    aabb_min = np.min(ctube_vertices.reshape(-1, 3), axis=0)
    aabb_max = np.max(ctube_vertices.reshape(-1, 3), axis=0)
    extent_aabb = aabb_max - aabb_min
    M = ctube_vertices.shape[0]
    N = ctube_vertices.shape[1]
    
    ax.plot(directrix[:, 0], directrix[:, 1], lw=2.0, c=curve_color, alpha=1.0, zorder=1)
    ax.plot(directrix[:, 0], directrix[:, 1], '.', c=curve_color, alpha=1.0, zorder=1, markersize=3)

    # Draw cross-sections
    for i in range(M):
        ax.fill(ctube_vertices[i, :, 0], ctube_vertices[i, :, 1], color=tube_color, alpha=0.2)

    # Draw centroids of cross-sections
    for i in range(M):
        ax.plot(ctube_vertices[i, :, 0].mean(), ctube_vertices[i, :, 1].mean(), '.', color=curve_color, markersize=3)

    # Add lines connecting consecutive cross-sections
    for i in range(M-1):
        for j in range(N-1):
            ax.plot([ctube_vertices[i, j, 0], ctube_vertices[i+1, j, 0]], [ctube_vertices[i, j, 1], ctube_vertices[i+1, j, 1]], c=tube_color, alpha=0.3, linewidth=0.5)
        # Make the last tube edge thicker
        j = N-1
        ax.plot([ctube_vertices[i, j, 0], ctube_vertices[i+1, j, 0]], [ctube_vertices[i, j, 1], ctube_vertices[i+1, j, 1]], c=tube_color, alpha=0.9, linewidth=1)

    # Draw target cross-sections in gray
    if target_cross_sections is not None:
        n_target_cross_sections = target_cross_sections.shape[0]
        for i in range(n_target_cross_sections):
            ax.fill(target_cross_sections[i, :, 0], target_cross_sections[i, :, 1], color=target_cross_section_color, alpha=0.5)

    ax.set_aspect('equal')
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim([aabb_min[0] - 0.05 * extent_aabb[0], aabb_max[0] + 0.05 * extent_aabb[0]])
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim([aabb_min[1] - 0.05 * extent_aabb[1], aabb_max[1] + 0.05 * extent_aabb[1]])
    ax.axis('off')

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax

def plot_tube_3d(directrix, ctube_vertices, cps=None, fig=None, ax=None, save_path=None, xlim=None, ylim=None, zlim=None, expand_axes=0.3, target_cross_sections=None, plot_curve_shadows=True, plot_planes=True, plot_origin=True, curve_color=None, tube_color=None, target_cross_section_color=None):
    if fig is None and ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
    else:
        assert fig is not None and ax is not None, "If fig and ax are provided, both must be provided."

    # Default colors
    if curve_color is None:
        curve_color = 'k'
    if tube_color is None:
        tube_color = 'C0'
    if target_cross_section_color is None:
        target_cross_section_color = 'r'

    # No gradients needed here: detach and convert to numpy
    if isinstance(directrix, torch.Tensor):
        directrix = directrix.detach().numpy()
    if isinstance(ctube_vertices, torch.Tensor):
        ctube_vertices = ctube_vertices.detach().numpy()
    if target_cross_sections is not None and isinstance(target_cross_sections, torch.Tensor):
        target_cross_sections = target_cross_sections.detach().numpy()

    # Plot curve
    ax.plot(directrix[:, 0], directrix[:, 1], directrix[:, 2], curve_color, label='optimized curve')
    ax.plot(*directrix[0], curve_color, marker='.', markersize=6)  # plot marker on the first and last nodes
    ax.plot(*directrix[-1], curve_color, marker='.', markersize=6)

    M = ctube_vertices.shape[0]
    N = ctube_vertices.shape[1]

    # Plot cross-sections
    for i in range(1, M-1):
        verts = [list(zip(ctube_vertices[i, :, 0], ctube_vertices[i, :, 1], ctube_vertices[i, :, 2]))]
        poly = Poly3DCollection(verts, color=tube_color, alpha=0.2, linewidth=LINEWIDTH_REF / 2.0)
        ax.add_collection3d(poly)
    # Use custom hatch pattern for the first and last cross-sections
    i = 0
    verts = [list(zip(ctube_vertices[i, :, 0], ctube_vertices[i, :, 1], ctube_vertices[i, :, 2]))]
    poly = Poly3DCollection(verts, color=tube_color, alpha=0.5, hatch='.', linewidth=LINEWIDTH_REF / 5)
    ax.add_collection3d(poly)
    i = M - 1
    verts = [list(zip(ctube_vertices[i, :, 0], ctube_vertices[i, :, 1], ctube_vertices[i, :, 2]))]
    poly = Poly3DCollection(verts, color=tube_color, alpha=0.5, hatch='o', linewidth=LINEWIDTH_REF / 5)
    ax.add_collection3d(poly)

    # Plot swept tube in 3d
    for i in range(M-1):
        for j in range(1, N):
            ridge = ctube_vertices[:, j, :] 
            ax.plot(*ridge.T, tube_color, alpha=0.2, linewidth=LINEWIDTH_REF / 2.0)
        # Make the first ridge thicker
        j = 0
        ridge = ctube_vertices[:, j, :]
        ax.plot(*ridge.T, tube_color, alpha=0.5, linewidth=LINEWIDTH_REF)

    # Plot target cross-sections
    if target_cross_sections is not None:
        for i in range(len(target_cross_sections)):
            verts = [list(zip(target_cross_sections[i, :, 0], target_cross_sections[i, :, 1], target_cross_sections[i, :, 2]))]
            poly = Poly3DCollection(verts, color=target_cross_section_color, alpha=0.5, linewidth=LINEWIDTH_REF)
            ax.add_collection3d(poly)

    if xlim is None and ylim is None and zlim is None:
        # Set equal aspect ratio
        max_range = np.max(np.max(ctube_vertices.reshape(-1, 3), axis=0) - np.min(ctube_vertices.reshape(-1, 3), axis=0))
        mid_point = 0.5 * (np.max(ctube_vertices.reshape(-1, 3), axis=0) + np.min(ctube_vertices.reshape(-1, 3), axis=0))
        xlim = [mid_point[0] - 0.5 * max_range - expand_axes * max_range, mid_point[0] + 0.5 * max_range + expand_axes * max_range]
        ylim = [mid_point[1] - 0.5 * max_range - expand_axes * max_range, mid_point[1] + 0.5 * max_range + expand_axes * max_range]
        zlim = [mid_point[2] - 0.5 * max_range - expand_axes * max_range, mid_point[2] + 0.5 * max_range + expand_axes * max_range]
    else:
        assert xlim is not None and ylim is not None and zlim is not None, "If any of xlim, ylim, zlim is provided, all must be provided."
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    # Plot origin
    if plot_origin:
        ax.plot(0.0, 0.0, 0.0, 'k+', markersize=6)

    # Plot spline control points
    if cps is not None:
        if isinstance(cps, torch.Tensor):
            cps = cps.detach().numpy()
        ax.plot(cps[:, 0], cps[:, 1], cps[:, 2], 'o', color='k', markersize=2)

    # Plot curve shadows by projecting points on xy-, yz-, and xz-planes
    if plot_curve_shadows:
        if plot_planes:
            directrix_xy, directrix_yz, directrix_xz = directrix.copy(), directrix.copy(), directrix.copy()
            directrix_xy[:, 2] = zlim[0]
            directrix_yz[:, 0] = xlim[0]
            directrix_xz[:, 1] = ylim[1]
            ax.plot(directrix_xy[:, 0], directrix_xy[:, 1], directrix_xy[:, 2], curve_color, alpha=0.6)
            ax.plot(directrix_yz[:, 0], directrix_yz[:, 1], directrix_yz[:, 2], curve_color, alpha=0.6)
            ax.plot(directrix_xz[:, 0], directrix_xz[:, 1], directrix_xz[:, 2], curve_color, alpha=0.6)
            # Define bounding planes, fill with gray, alpha = 0.1
            verts = [
                [(xlim[0], ylim[0], zlim[0]), (xlim[1], ylim[0], zlim[0]), (xlim[1], ylim[1], zlim[0]), (xlim[0], ylim[1], zlim[0])],
                [(xlim[0], ylim[0], zlim[0]), (xlim[0], ylim[1], zlim[0]), (xlim[0], ylim[1], zlim[1]), (xlim[0], ylim[0], zlim[1])],
                [(xlim[0], ylim[1], zlim[0]), (xlim[1], ylim[1], zlim[0]), (xlim[1], ylim[1], zlim[1]), (xlim[0], ylim[1], zlim[1])]
            ]
            poly = Poly3DCollection(verts, color='gray', alpha=0.05, linewidth=0.05)
            ax.add_collection3d(poly)

        # Plot ridge shadows by projecting points on xy-, yz-, and xz-planes
        j = 0
        ridge = ctube_vertices[:, j, :]
        ridge_xy, ridge_yz, ridge_xz = ridge.copy(), ridge.copy(), ridge.copy()
        ridge_xy[:, 2] = zlim[0]
        ridge_yz[:, 0] = xlim[0]
        ridge_xz[:, 1] = ylim[1]
        ax.plot(ridge_xy[:, 0], ridge_xy[:, 1], ridge_xy[:, 2], tube_color, alpha=0.6, linewidth=LINEWIDTH_REF)
        ax.plot(ridge_yz[:, 0], ridge_yz[:, 1], ridge_yz[:, 2], tube_color, alpha=0.6, linewidth=LINEWIDTH_REF)
        ax.plot(ridge_xz[:, 0], ridge_xz[:, 1], ridge_xz[:, 2], tube_color, alpha=0.6, linewidth=LINEWIDTH_REF)

    ax.axis('off')
    ax.view_init(elev=30, azim=-60)

    # plt.show()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax

def plot_unrolled_strips(ctube_vertices, fig=None, ax=None, save_path=None, xlim=None, ylim=None, y_offset_total=None, y_offset_per_strip=None, selected_strips=None, axes_first_edges=None, points_first_nodes=None, strip_colors=None, offscreen=False):
    '''
    Returns:
        fig, ax
    '''
    
    if fig is None and ax is None:
        fig = plt.figure(figsize=(6, 6), dpi=100)
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0, 0])
    else:
        assert fig is not None and ax is not None, "If fig and ax are provided, both must be provided."
    
    if offscreen:
        plt.ioff()  # turn off interactive mode (no display)

    lw = 0.5

    M = ctube_vertices.shape[0]
    N = ctube_vertices.shape[1]

    vertices_per_strip, faces_per_strip = compute_unrolled_strips(
        ctube_vertices, 
        y_offset_total=y_offset_total, 
        y_offset_per_strip=y_offset_per_strip, 
        selected_strips=selected_strips, 
        axes_first_edges=axes_first_edges, 
        points_first_nodes=points_first_nodes
    )

    # Plot
    n_strips = len(selected_strips) if selected_strips is not None else N
    if strip_colors is None:
        strip_colors = ['C0' for i in range(n_strips)]
    else:
        assert len(strip_colors) == n_strips, "The number of colors must match the number of strips, but got {} colors for {} strips.".format(len(strip_colors), n_strips)
    for strip_idx in range(n_strips):
        strip_vertices = vertices_per_strip[strip_idx].detach().numpy()
        strip_faces = faces_per_strip[strip_idx].detach().numpy()
        boundary_edges, interior_edges = extract_edges_from_faces(strip_faces)  # different plotting style for boundary and interior edges
        for ei, edge in enumerate(boundary_edges):
            if ei == 2:  # Plot arrow for the initial edge (which is indexed by 2 given the connectivity used by get_flattened_strips)
                ax.quiver(strip_vertices[edge[0], 0], strip_vertices[edge[0], 1], strip_vertices[edge[1], 0] - strip_vertices[edge[0], 0], strip_vertices[edge[1], 1] - strip_vertices[edge[0], 1], color=strip_colors[strip_idx], scale=1, scale_units='xy', width=0.005, headwidth=3, headlength=3, headaxislength=3)
            else:
                ax.plot(strip_vertices[edge, 0], strip_vertices[edge, 1], lw=lw, c=strip_colors[strip_idx], zorder=0)
        for edge in interior_edges:
            ax.plot(strip_vertices[edge, 0], strip_vertices[edge, 1], lw=lw/2, linestyle='--', alpha=0.5, c=strip_colors[strip_idx], zorder=0)
            
    ax.set_aspect('equal')
    ax.axis('off')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if offscreen:
        plt.close(fig)
        plt.ion()

    return fig, ax

def plot_generatrix(pts_cross_section, fig=None, ax=None, save_path=None, xlim=None, ylim=None, curve_color='k', offscreen=False, polygonal=True, plot_origin=True):
    if fig is None and ax is None:
        fig = plt.figure(figsize=(4, 4), dpi=100)
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0, 0])
    else:
        assert fig is not None and ax is not None, "If fig and ax are provided, both must be provided."

    # No gradients needed here: detach and convert to numpy
    if isinstance(pts_cross_section, torch.Tensor):
        pts_cross_section = pts_cross_section.detach().numpy()
    
    if polygonal:
        polygon = Polygon(pts_cross_section[:, :2], closed=True, edgecolor='none', facecolor=curve_color, lw=LINEWIDTH_REF, alpha=0.1)
        ax.add_patch(polygon)
        polygon = Polygon(pts_cross_section[:, :2], closed=True, edgecolor=curve_color, facecolor='none', lw=LINEWIDTH_REF, alpha=1.0)
        ax.add_patch(polygon)
        ax.scatter(pts_cross_section[:, 0], pts_cross_section[:, 1], c=curve_color, alpha=1.0, zorder=1, s=20)

    else:
        ax.plot(pts_cross_section[:, 0], pts_cross_section[:, 1], lw=LINEWIDTH_REF, c=curve_color, alpha=1.0, zorder=1)
        ax.scatter(pts_cross_section[:, 0], pts_cross_section[:, 1], c=curve_color, alpha=1.0, zorder=1, s=20)

    if plot_origin:
        ax.plot([0], [0], 'k+', markersize=10, markeredgewidth=2)

    ax.set_aspect('equal')
    ax.axis('off')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    if offscreen:
        plt.close(fig)
        plt.ion()

    return fig, ax

def plot_convex_hull(pts, fig=None, ax=None, save_path=None, xlim=None, ylim=None, zlim=None, facecolor='k', alpha=1.0, offscreen=False):
    if fig is None and ax is None:
        fig = plt.figure(figsize=(8, 8), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
    else:
        assert fig is not None and ax is not None, "If fig and ax are provided, both must be provided."

    # No gradients needed here: detach and convert to numpy
    if isinstance(pts, torch.Tensor):
        pts = pts.detach().numpy()

    hull = ConvexHull(pts)
    ax.plot_trisurf(pts[:, 0], pts[:, 1], pts[:, 2], triangles=hull.simplices, facecolor=facecolor, edgecolor=[[0, 0, 0]], linewidth=LINEWIDTH_REF / 2.0, alpha=alpha, shade=True)
    
    ax.set_aspect('equal')
    ax.axis('off')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_zlim(zlim)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    if offscreen:
        plt.close(fig)
        plt.ion()

    return fig, ax

# --------------------------------------------------------------------------------
# Animations
# --------------------------------------------------------------------------------

def render_video(file_name_prefix, output_file, path_to_output, fps=25, loop=1):
    import ffmpeg
    
    # Extract png size in pixels
    im = Image.open(os.path.join(path_to_output, file_name_prefix + "{:04d}.png".format(0)))  # assume all images are the same size
    width, height = im.size
    # Render video
    (
        ffmpeg
        .input(os.path.join(path_to_output, file_name_prefix + "*.png"), pattern_type='glob', framerate=fps)
        .output(os.path.join(path_to_output, output_file), vcodec='libopenh264', pix_fmt='yuv420p', vf="color=white:{}x{} [bg]; [bg][0:v] overlay=shortest=1".format(width, height))
        .run(overwrite_output=True)
    )
    if loop > 1:
        # Loop the video using ffmpeg
        input_file = os.path.join(path_to_output, output_file)
        output_file_looped = os.path.join(path_to_output, "looped_" + output_file)
        (
            ffmpeg
            .input(input_file, stream_loop=loop-1)
            .output(output_file_looped, vcodec='libx264', pix_fmt='yuv420p', vf="color=white:{}x{} [bg]; [bg][0:v] overlay=shortest=1".format(width, height))
            .run(overwrite_output=True)
        )
        os.remove(input_file)  # remove the original video
        os.rename(output_file_looped, input_file)  # rename the looped video to the original name
