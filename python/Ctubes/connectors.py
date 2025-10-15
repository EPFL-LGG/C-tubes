import os
import sys as _sys

PATH_TO_SCRIPT = os.path.dirname(os.path.realpath(__file__))
PATH_TO_PYTHON = os.path.dirname(PATH_TO_SCRIPT)
_sys.path += [PATH_TO_PYTHON]

import torch
from Ctubes.geometry_utils import rotate_about_axis, rotate_about_axes, matrix_to_axis_angle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from Ctubes.plot_utils import plot_generatrix, plot_convex_hull, LINEWIDTH_REF
from scipy.spatial import ConvexHull

class Arm:
    """
    A class to represent the arm of a joint.
    """
    def __init__(self, cross_section_vertices, twist_angle):
        """
        Args:
            cross_section_vertices (torch tensor of shape (n_cs, 2)): The vertices of the cross section of the arm.
            twist_angle (torch tensor of shape ()): The angle of twist of the arm.
        """
        if not isinstance(cross_section_vertices, torch.Tensor):
            cross_section_vertices = torch.tensor(cross_section_vertices)
        if not isinstance(twist_angle, torch.Tensor):
            twist_angle = torch.tensor(twist_angle)
        self.cross_section_vertices = cross_section_vertices
        self.cross_section_vertices_3d = torch.cat([cross_section_vertices, torch.zeros(cross_section_vertices.shape[0], 1)], dim=1)
        self.twist_angle = twist_angle
        self.N = cross_section_vertices.shape[0]
        
    def compute_aabb(self):
        vertices = self.get_cross_section_vertices_3d()
        aabb_min = torch.min(vertices, dim=0).values
        aabb_max = torch.max(vertices, dim=0).values
        return aabb_min, aabb_max
    
    def aabb_diagonal_length(self):
        aabb_min, aabb_max = self.compute_aabb()
        return torch.linalg.norm(aabb_max - aabb_min)
        
    def get_cross_section_vertices_3d(self):
        return rotate_about_axis(self.cross_section_vertices_3d, torch.tensor([0.0, 0.0, 1.0]), self.twist_angle)
    
    def get_cross_section_vertices(self):
        return rotate_about_axis(self.cross_section_vertices_3d, torch.tensor([0.0, 0.0, 1.0]), self.twist_angle)[..., :2]
    
    def get_geometry(self):
        pts = self.cross_section_vertices_3d.numpy()
        if pts.shape[0] == 3:
            triangles = [[0, 1, 2]]
        else:
            hull = ConvexHull(pts)
            triangles = hull.simplices.tolist()
        return {
            'vertices': pts.tolist(),
            'twist_angle': self.twist_angle.item(),
            'triangles': triangles,
            'faces': [[i for i in range(self.N)]],
        }
    
    def plot(self, fig=None, ax=None, save_path=None, xlim=None, ylim=None, offscreen=False):
        return plot_generatrix(self.get_cross_section_vertices(), fig=fig, ax=ax, save_path=save_path, xlim=xlim, ylim=ylim, offscreen=offscreen)

class Connector:
    def __init__(self, arms, arm_offsets, arm_orientations, position, orientation):
        '''
        Args:
            arms (list of Arm): The arms of the connection.
            arm_offsets (torch tensor of shape (n_arms, 3)): The offsets of the arms.
            arm_orientations (torch tensor of shape (n_arms, 3)): the axis angle of rotation that transforms the xyz frame into the arm frame, z is transformed to the arm direction.
            position (torch tensor of shape (3,)): The position of the connector.
            orientation (torch tensor of shape (3,)): gives the axis of rotation that transforms the the xyz frame into the connector frame.
        '''
        assert all([isinstance(arm, Arm) for arm in arms]), "All arms must be of type Arm."
        self.connector_name = "Connector"
        self.arms = arms
        self.arm_offsets = arm_offsets
        self.arm_orientations = arm_orientations
        self.position = position
        self.orientation = orientation
        self.n_arms = len(arms)
        self.N_per_arm = [arm.N for arm in arms]
        
        self.connector_frame = torch.eye(3)
        self.update_angle_and_axis()
        
    def update_angle_and_axis(self):
        self.orientation_angle = torch.linalg.norm(self.orientation) + 1.0e-8
        self.orientation_axis = self.orientation / self.orientation_angle
        
        self.arm_orientation_angles = torch.linalg.norm(self.arm_orientations, dim=1) + 1.0e-8
        self.arm_orientation_axes = self.arm_orientations / self.arm_orientation_angles.unsqueeze(1)
        
    def set_position(self, position):
        self.position = position
        
    def set_orientation(self, orientation):
        self.orientation = orientation
        self.update_angle_and_axis()
        
    def get_position(self):
        return self.position
    
    def get_orientation(self):
        return self.orientation
    
    def compute_aabb(self):
        vertices = torch.cat([arm.get_cross_section_vertices_3d() for arm in self.arms], dim=0)
        aabb_min = torch.min(vertices, dim=0).values
        aabb_max = torch.max(vertices, dim=0).values
        return aabb_min, aabb_max
    
    def aabb_diagonal_length(self):
        aabb_min, aabb_max = self.compute_aabb()
        return torch.linalg.norm(aabb_max - aabb_min)
        
    def transform_connector_to_world_frame(self, vertices):
        '''
        Args:
            vertices (torch tensor of shape (n_vertices, 3)): The vertices to transform.
            
        Returns:
            torch tensor of shape (n_vertices, 3): The vertices transformed to the world frame.
        '''
        return rotate_about_axis(vertices, self.orientation_axis, self.orientation_angle) + self.position.unsqueeze(0)
        
    def transform_world_to_connector_frame(self, vertices):
        '''
        Args:
            vertices (torch tensor of shape (n_vertices, 3)): The vertices to transform.
            
        Returns:
            torch tensor of shape (n_vertices, 3): The vertices transformed to the connector frame.
        '''
        
        return rotate_about_axis(vertices - self.position.unsqueeze(0), self.orientation_axis, -self.orientation_angle)
    
    def transform_arm_frame_to_world(self, vertices, arm_id):
        '''
        Args:
            vertices (torch tensor of shape (n_vertices, 3)): The vertices to transform expressed in the arm frame.
            arm_id (torch.tensor of shape (n_vertices,)): The id of the arm to transform to for each vertex.
            
        Returns:
            torch tensor of shape (n_vertices, 3): The vertices expressed in the world coordinate system.
        '''
        rotated_vertices_to_connector_frame = rotate_about_axes(vertices, self.arm_orientation_axes[arm_id], self.arm_orientation_angles[arm_id])
        
        return self.transform_connector_to_world_frame(rotated_vertices_to_connector_frame + self.arm_offsets[arm_id])
    
    def transform_world_to_arm_frame(self, vertices, arm_id):
        '''
        Args:
            vertices (torch tensor of shape (n_vertices, 3)): The vertices to transform expressed in the world frame.
            arm_id (torch.tensor of shape (n_vertices,)): The id of the arm to transform to for each vertex.
            
        Returns:
            torch tensor of shape (n_vertices, 3): The vertices transformed to their respective arm frame.
        '''
        rotated_vertices_to_connector_frame = self.transform_world_to_connector_frame(vertices)
        
        return rotate_about_axes(rotated_vertices_to_connector_frame - self.arm_offsets[arm_id], self.arm_orientation_axes[arm_id], -self.arm_orientation_angles[arm_id])
    
    def get_arm_cross_section_vertices(self, arm_id):
        '''
        Args:
            arm_id (int): The id of the arm.
        
        Returns:
            torch tensor of shape (n_cs, 2): The cross section vertices of one arm in the world frame.
        '''
        assert arm_id < self.n_arms, "The arm_id must be less than the number of arms."
        arms_ids = arm_id * torch.ones(self.arms[arm_id].N, dtype=torch.int64)
        return self.transform_arm_frame_to_world(self.arms[arm_id].get_cross_section_vertices_3d(), arms_ids)
    
    def get_geometry(self):
        pts = self.get_arms_cross_section_vertices().reshape(-1, 3)
        if pts.shape[0] == 3:
            triangles = [[0, 1, 2]]
        else:
            hull = ConvexHull(pts.detach().numpy())
            triangles = hull.simplices.tolist()
        pts_per_arm = [self.get_arm_cross_section_vertices(arm_id).detach().numpy().tolist() for arm_id in range(self.n_arms)]
        return {
            'name': self.connector_name,
            'vertices': pts.detach().numpy().tolist(),
            'vertices_per_arm': pts_per_arm,
            'triangles': triangles,
            'arms': [arm.get_geometry() for arm in self.arms],
            'position': self.position.detach().numpy().tolist(),
            'orientation': self.orientation.detach().numpy().tolist(),
        }
    
    def get_arms_cross_section_vertices(self):
        '''
        Returns:
            torch tensor of shape (n_arms, n_cs, 3): The cross section vertices of each arm in the world space.
        '''
        arm_ids = torch.repeat_interleave(torch.arange(self.n_arms), repeats=torch.tensor([arm.N for arm in self.arms]), dim=0)
        return self.transform_arm_frame_to_world(torch.cat([arm.get_cross_section_vertices_3d() for arm in self.arms], dim=0), arm_ids).reshape(self.n_arms, -1, 3)
    
    def plot(self, fig=None, ax=None, save_path=None, xlim=None, ylim=None, zlim=None, facecolor=(0.5, 0.5, 0.5, 0.5), alpha=1.0, arm_section_color='r', offscreen=False, show_arm_faces=False):
        # print(self.get_arms_cross_section_vertices().shape)
        fig, ax = plot_convex_hull(self.get_arms_cross_section_vertices().reshape(-1, 3), fig=fig, ax=ax, save_path=save_path, xlim=xlim, ylim=ylim, zlim=zlim, facecolor=facecolor, alpha=alpha, offscreen=offscreen)
        if show_arm_faces:
            verts = list(self.get_arms_cross_section_vertices().detach().numpy()) # add a list of arrays of shape (n_points_cs, 3)
            poly = Poly3DCollection(verts, color=arm_section_color, alpha=0.5, linewidth=LINEWIDTH_REF)
            ax.add_collection3d(poly)
        return fig, ax
    
class FlatConnector(Connector):
    
    def __init__(self, arms, arm_angles, position, orientation, radius):
        self.connector_name = "FlatConnector"
        
        assert all([isinstance(arm, Arm) for arm in arms]), "All arms must be of type Arm."
        
        if not isinstance(radius, torch.Tensor):
            radius = torch.tensor(radius)
        
        self.n_arms = len(arms)
        self.radius = radius
        self.arm_angles = arm_angles
        arm_offsets = torch.zeros(self.n_arms, 3)
        arm_offsets[:, 0] = self.radius * torch.cos(arm_angles)
        arm_offsets[:, 1] = self.radius * torch.sin(arm_angles)
        
        self.arm_frames_in_connector_frame = torch.zeros(size=(self.n_arms, 3, 3))
        # z axis of the arm frame is the arm direction.
        self.arm_frames_in_connector_frame[:, :, 2] = arm_offsets / self.radius 
        # y axis of the arm frame is the z axis of the connector frame
        self.arm_frames_in_connector_frame[:, 2, 1] = 1.0
        self.arm_frames_in_connector_frame[:, :, 0] = torch.cross(self.arm_frames_in_connector_frame[:, :, 1], self.arm_frames_in_connector_frame[:, :, 2], dim=1)
        
        arm_orientation = matrix_to_axis_angle(self.arm_frames_in_connector_frame)
        
        super().__init__(arms, arm_offsets, arm_orientation, position, orientation)
        
    def get_geometry(self):
        geom = super().get_geometry()
        geom['radius'] = self.radius.item()
        geom['arm_angles'] = self.arm_angles.numpy().tolist()
        return geom
    
class FlatRegularConnector(FlatConnector):
    
    def __init__(self, arms, position, orientation, radius):
        self.connector_name = "FlatRegularConnector"
        arm_angles = torch.linspace(0.0, 2.0 * torch.pi, len(arms)+1)[:-1]
        super().__init__(arms, arm_angles, position, orientation, radius)
        
# The scaling factors scales a regular tetrahedron with edge length 1.
# When the arms are composed of equilateral triangles (tube radius of sqrt(3)/3) appropriately twisted, the connector is a regular tetrahedron.
class TetrahedralConnector(Connector):
    
    def __init__(self, arms, position, orientation, scale):
        self.connector_name = "TetrahedralConnector"
        assert all([isinstance(arm, Arm) for arm in arms]), "All arms must be of type Arm."
        
        self.n_arms = len(arms)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale)
        assert self.n_arms==4, "There must be exactly four arms to assign to the tetrahedral connector."
        
        height = torch.sqrt(torch.tensor(2.0/3.0))
        dihedral_angle = torch.acos(torch.tensor(1.0/3.0))
        self.scale = scale
        arm_offsets = torch.zeros(4, 3)
        # The angle of the three face normas with the z axis is pi-dihedral_angle.
        arm_offsets[:3, 0] = torch.sin(dihedral_angle) * self.scale * torch.cos(torch.linspace(0.0, 2.0 * torch.pi, 4)[:-1])
        arm_offsets[:3, 1] = torch.sin(dihedral_angle) * self.scale * torch.sin(torch.linspace(0.0, 2.0 * torch.pi, 4)[:-1])
        arm_offsets[:3, 2] = - torch.cos(dihedral_angle) * self.scale
        arm_offsets[3, 2] = self.scale
        arm_offsets *= height / 4.0 # the inradius of a regular tetrahedron
        
        self.arm_frames_in_connector_frame = torch.zeros(size=(self.n_arms, 3, 3))
        # z axis of the arm frame is the arm direction.
        self.arm_frames_in_connector_frame[:, :, 2] = arm_offsets / torch.linalg.norm(arm_offsets, dim=1, keepdim=True)
        # y axis of the arm frame is the z axis rotated by "pi/2 - dihedral_angle" degrees in the direction of the offset, except for the top arm.
        self.arm_frames_in_connector_frame[:3, 0, 1] = torch.cos(dihedral_angle) * torch.cos(torch.linspace(0.0, 2.0 * torch.pi, 4)[:-1])
        self.arm_frames_in_connector_frame[:3, 1, 1] = torch.cos(dihedral_angle) * torch.sin(torch.linspace(0.0, 2.0 * torch.pi, 4)[:-1])
        self.arm_frames_in_connector_frame[:3, 2, 1] = torch.sin(dihedral_angle)
        self.arm_frames_in_connector_frame[3, 0, 1] = 1.0
        
        self.arm_frames_in_connector_frame[:, :, 0] = torch.cross(self.arm_frames_in_connector_frame[:, :, 1], self.arm_frames_in_connector_frame[:, :, 2], dim=1)
        
        arm_orientation = matrix_to_axis_angle(self.arm_frames_in_connector_frame)
        
        super().__init__(arms, arm_offsets, arm_orientation, position, orientation)
        
