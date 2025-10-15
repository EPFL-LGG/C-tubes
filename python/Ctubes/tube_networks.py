from Ctubes.connectors import Connector
from Ctubes.tubes import CTube
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from Ctubes.misc_utils import write_polyline_to_obj, write_mesh_to_obj

# Wrapper class around multiple CTube objects.
# The class stores a list of CTube objects and provides methods to set the parameters of all tubes at once.
class CTubeNetwork:
    def __init__(self, tubes, connectors, connector_to_tube_connectivity):
        """
        Args:
            tubes (list of CTube): List of CTube objects that form the tube network.
            connectors (list of Connector): List of Connector objects that connect the tubes.
            connector_to_tube_connectivity (list of list of pairs of int): List of lists of pairs of integers that specify the connectivity between connectors and tubes. The pair P located in [i][j] specifies the connector i is connected through arm j of the connector to the tube P[0] at the start (P[1] == 0) or end (P[1] == 1).
        """
        assert all([isinstance(tube, CTube) for tube in tubes]), "All tubes must be of type CTube."
        assert all([isinstance(connector, Connector) for connector in connectors]), "All connectors must be of type Connector."
        self.tubes = tubes
        self.n_tubes = len(tubes)
        self.M_per_tube = [tube.directrix.M for tube in tubes]
        self.K_per_tube = [tube.directrix.K for tube in tubes]
        self.N_per_tube = [tube.generatrix.N for tube in tubes]
        self.M = sum(self.M_per_tube)
        self.K = sum(self.K_per_tube)
        self.n_closed_curves = sum([tube.directrix.closed_curve for tube in tubes])
        self.n_tubes_with_symmetry = sum([tube.has_symmetry() for tube in tubes])
        self.n_tubes_with_planar_profile = sum([tube.has_planar_profile() for tube in tubes])
        
        self.connectors = connectors
        self.n_arms = sum([connector.n_arms for connector in connectors])
        self.N_per_connector = [connector.N_per_arm for connector in connectors]
        self.ctt_connectivity = connector_to_tube_connectivity
        self.n_connectors = len(connectors)
        self.ttc_connectivity = None
        self.infer_tube_to_connector_connectivity()

    def has_symmetry(self):
        return self.n_tubes_with_symmetry > 0
    
    def has_planar_profile(self):
        return self.n_tubes_with_planar_profile > 0
        
    def infer_tube_to_connector_connectivity(self):
        '''
        tube_to_connector_connectivity (list of pairs of pairs of int): List of pairs of pairs of integers that specify the connectivity between tubes and connectors. Each element of the list corresponds to one tube: the first pair specifies the connector ID and the arm ID at the start of the tube, the second pair specifies the connector ID and the arm ID at the end of the tube.
        '''
        self.ttc_connectivity = [[None, None] for _ in range(self.n_tubes)]
        for id_c, c_tmp in enumerate(self.ctt_connectivity):
            for id_a, a_tmp in enumerate(c_tmp):
                self.ttc_connectivity[a_tmp[0]][a_tmp[1]] = [id_c, id_a]
                
    def print_connector_to_tube_connectivity(self):
        for id_c, c_tmp in enumerate(self.ctt_connectivity):
            for id_a, a_tmp in enumerate(c_tmp):
                start_tag = "start" if a_tmp[1] == 0 else "end"
                print(f"Connector {id_c} is connected to tube {a_tmp[0]} at the {start_tag}")

    def get_polylines(self):
        directrixs = []
        for tube in self.tubes:
            directrixs.append(tube.get_polyline())
        return directrixs
    
    def get_polylines_with_symmetry(self):
        directrixs = []
        for tube in self.tubes:
            directrixs.append(tube.get_polyline_with_symmetry())
        return directrixs

    def compute_vertices(self):
        ctube_vertices = []
        for tube in self.tubes:
            ctube_vertices.append(tube.compute_vertices())
        return ctube_vertices
    
    def update_ctube_vertices_per_tube(self):
        for tube in self.tubes:
            tube.update_ctube_vertices()

    def compute_aabb(self):
        ctube_vertices = torch.cat([sw.reshape(-1, 3) for sw in self.compute_vertices()], dim=0)
        aabb_min = torch.min(ctube_vertices, dim=0).values
        aabb_max = torch.max(ctube_vertices, dim=0).values
        return aabb_min, aabb_max
    
    def aabb_diagonal_length(self):
        aabb_min, aabb_max = self.compute_aabb()
        return torch.linalg.norm(aabb_max - aabb_min)

    def set_control_points(self, cps):
        assert cps.shape[0] == self.K, "The number of control points ({}) must match the total number of control points of all tubes ({}).".format(cps.shape[0], self.K)
        idx = 0
        for i, tube in enumerate(self.tubes):
            tube.set_control_points(cps[idx:idx+tube.directrix.K])
            idx += tube.directrix.K

    def set_thetas(self, thetas):
        assert thetas.numel() == self.n_tubes, "The number of initial angles ({}) must match the number of tubes ({}).".format(thetas.numel(), self.n_tubes)
        for i, tube in enumerate(self.tubes):
            tube.set_theta(thetas[i])

    def set_apex_loc_func(self, apex_loc_func):
        assert apex_loc_func.numel() == self.M - self.n_tubes, "The number of apex locating function ({}) must match the number of discretization points of all tubes minus the number of tubes ({}).".format(apex_loc_func.numel(), self.M - self.n_tubes)
        idx = 0
        for i, tube in enumerate(self.tubes):
            tube.set_apex_loc_func(apex_loc_func[idx:idx+tube.M-1])
            idx += tube.M - 1

    def set_plane_normals(self, plane_normals):
        if self.has_planar_profile():
            assert plane_normals.shape[0] == self.M, "The number of plane normals ({}) must match the number of discretization points of all tubes ({}).".format(plane_normals.shape[0], self.M)
            idx = 0
            for i, tube in enumerate(self.tubes):
                tube.set_plane_normals(plane_normals[idx:idx+tube.M])
                idx += tube.M
        else:
            assert self.n_tubes == 1, "Only one tube with non-planar profile is supported."
            tube = self.tubes[0]
            assert plane_normals.shape[0] == tube.M
            assert plane_normals.shape[1] == tube.N
            tube.set_plane_normals(plane_normals)

    def set_init_scales(self, init_scales):
        assert init_scales.numel() == self.n_tubes, "The number of initial scales ({}) must match the number of tubes ({}).".format(init_scales.numel(), self.n_tubes)
        for i, tube in enumerate(self.tubes):
            tube.set_init_scale(init_scales[i])

    def set_dihedral_angles(self, dihedral_angles):
        assert isinstance(dihedral_angles, list), "Dihedral angles must be provided as a list (one tensor per tube)."  # we cannot stack them as they might have different size in both the first dimensions (M and N)
        assert [angles.numel() for angles in dihedral_angles] == [(self.M_per_tube[i]-1) * self.N_per_tube[i] for i in range(self.n_tubes)], "The number of dihedral angles ({}) must match the number of discretization points - 1 times the number of cross-sections of all tubes ({}).".format([angles.numel() for angles in dihedral_angles], [self.M_per_tube[i]-1 * self.N_per_tube[i] for i in range(self.n_tubes)])
        for i, tube in enumerate(self.tubes):
            tube.set_dihedral_angles(dihedral_angles[i])

    def set_connector_positions(self, positions):
        assert positions.shape[0] == self.n_connectors, "The number of connector positions ({}) must match the number of connectors ({}).".format(positions.shape[0], self.n_connectors)
        for i, connector in enumerate(self.connectors):
            connector.set_position(positions[i])
            
    def set_connector_orientations(self, orientations):
        assert orientations.shape[0] == self.n_connectors, "The number of connector orientations ({}) must match the number of connectors ({}).".format(orientations.shape[0], self.n_connectors)
        for i, connector in enumerate(self.connectors):
            connector.set_orientation(orientations[i])

    def get_control_points(self):
        cps = torch.zeros(size=(self.K, 3))
        idx = 0
        for i, tube in enumerate(self.tubes):
            cps[idx:idx+tube.directrix.K] = tube.get_control_points()
            idx += tube.directrix.K
        return cps
    
    def get_thetas(self):
        thetas = torch.zeros(size=(self.n_tubes,))
        for i, tube in enumerate(self.tubes):
            thetas[i] = tube.get_theta()
        return thetas
    
    def get_apex_loc_func(self):
        apex_loc_func = torch.zeros(size=(self.M - self.n_tubes,))
        idx = 0
        for i, tube in enumerate(self.tubes):
            apex_loc_func[idx:idx+tube.M-1] = tube.get_apex_loc_func()
            idx += tube.M - 1
        return apex_loc_func
    
    def get_plane_normals(self):
        if self.has_planar_profile():
            plane_normals = torch.zeros(size=(self.M, 3))
            idx = 0
            for i, tube in enumerate(self.tubes):
                plane_normals[idx:idx+tube.M] = tube.get_plane_normals()
                idx += tube.M
        else:
            assert self.n_tubes == 1, "Only one tube with nonplanar profile is supported."
            tube = self.tubes[0]
            plane_normals = tube.get_plane_normals() # (tube.M, tube.N, 3)
        return plane_normals
    
    def get_init_scales(self):
        init_scales = torch.zeros(size=(self.n_tubes,))
        for i, tube in enumerate(self.tubes):
            init_scales[i] = tube.get_init_scale()
        return init_scales
    
    def get_dihedral_angles(self):
        dihedral_angles = [tube.get_dihedral_angles() for tube in self.tubes]
        return dihedral_angles
    
    def get_connector_positions(self):
        positions = torch.zeros(size=(self.n_connectors, 3))
        for i, connector in enumerate(self.connectors):
            positions[i] = connector.get_position()
        return positions
    
    def get_connector_orientations(self):
        orientations = torch.zeros(size=(self.n_connectors, 3))
        for i, connector in enumerate(self.connectors):
            orientations[i] = connector.get_orientation()
        return orientations
    
    def get_plane_normal_kinds(self):
        plane_normal_kinds = []
        for i, tube in enumerate(self.tubes):
            plane_normal_kinds.append(tube.plane_normal_kind)
        return plane_normal_kinds
    
    def get_geometry(self):
        return {
            'tube_geometries': [tube.get_geometry() for tube in self.tubes],
            'connector_geometries': [connector.get_geometry() for connector in self.connectors],
        }
    
    def get_mesh(self, triangulate=False, with_symmetry=True, end_caps=False, split_strips=False):
        if self.n_tubes > 1:
            raise NotImplementedError("Getting the mesh of tube networks with more than one tube is not implemented yet.")
        return self.tubes[0].get_mesh(triangulate=triangulate, with_symmetry=with_symmetry, end_caps=end_caps, split_strips=split_strips)
        # if self.n_tubes > 1:
        #     raise NotImplementedError("Getting the mesh of tube networks with more than one tube is not implemented yet.")
        # # vertices = []
        # # faces = []
        # # for tube in self.tubes:
        # #     tube_vertices, tube_faces = tube.get_mesh(triangulate=triangulate, with_symmetry=with_symmetry, end_caps=end_caps, split_strips=split_strips)
        # #     vertices.append(tube_vertices)
        # #     faces.append(tube_faces)
        # tube_vertices, tube_faces = self.tubes[0].get_mesh(triangulate=triangulate, with_symmetry=with_symmetry, end_caps=end_caps, split_strips=split_strips)
        # return np.concatenate(vertices, axis=0), np.concatenate(faces, axis=0)
    
    def save_obj(self, file_path, triangulate=False, with_symmetry=True, end_caps=False, save_directrix=False, split_strips=False, max_quads_per_strip=None, object_name='tube_network'):
        if self.n_tubes == 1:
            self.tubes[0].save_obj(file_path, triangulate=triangulate, with_symmetry=with_symmetry, end_caps=end_caps, save_directrix=save_directrix, split_strips=split_strips, max_quads_per_strip=max_quads_per_strip)
        else:
            assert not split_strips, "Saving tube networks with split strips is not implemented yet."

            if os.path.exists(file_path):
                os.remove(file_path)

            # One mesh per tube
            n_vertices_tube_mesh = 0
            for i, tube in enumerate(self.tubes):
                V_tube, F_tube = tube.get_mesh(triangulate=triangulate, with_symmetry=with_symmetry, end_caps=end_caps, split_strips=split_strips)
                V = V_tube
                F = F_tube + n_vertices_tube_mesh
                n_vertices_tube_mesh += V.shape[0]
                object_name_tube = object_name + f"_tube_{i}" if object_name is not None else None
                assert isinstance(V, np.ndarray) and isinstance(F, np.ndarray), "The vertices and faces must be numpy arrays."
                write_mesh_to_obj(file_path, V, F, mode='a', object_name=object_name_tube)

            if save_directrix:
                for i, tube in enumerate(self.tubes):
                    directrix_name = object_name + f"_directrix_{i}" if object_name is not None else None
                    V = tube.get_polyline_with_symmetry().detach().numpy() if with_symmetry else tube.get_polyline().detach().numpy()
                    closed_curve = tube.directrix.closed_curve if not with_symmetry else False
                    write_polyline_to_obj(file_path, V, write_edges=True, edge_idx_offset=n_vertices_tube_mesh, mode='a', object_name=directrix_name, closed_curve=closed_curve)
                    n_vertices_tube_mesh += V.shape[0]

    def save_json(self, file_path):
        if self.n_tubes == 1:
            self.tubes[0].save_json(file_path)
        else:
            import json
            geo_dict = {}
            for i, tube in enumerate(self.tubes):
                geo_dict_tube = tube.get_geometry()
                geo_dict[f'tube_{i}'] = geo_dict_tube
            with open(file_path, 'w') as f:
                json.dump(geo_dict, f, indent=4)

    def plot(self, fig=None, ax=None, save_path=None, xlim=None, ylim=None):
        if fig is None and ax is None:
            fig = plt.figure(figsize=(8, 8), dpi=100)
            gs = fig.add_gridspec(1, 1)
            ax = fig.add_subplot(gs[0, 0])
        else:
            assert fig is not None and ax is not None, "If fig and ax are provided, both must be provided."
        for tube in self.tubes:
            fig, ax = tube.plot(fig=fig, ax=ax, save_path=None, xlim=xlim, ylim=ylim)
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig, ax
    
    def plot_3d(self, fig=None, ax=None, save_path=None, xlim=None, ylim=None, zlim=None, expand_axes=0.3, plot_curve_shadows=True, connector_face_color=(0.5, 0.5, 0.5, 0.5), connector_alpha=0.3, tube_colors=None, with_symmetry=False, show_arm_faces=False):
        if fig is None and ax is None:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection='3d')
        else:
            assert fig is not None and ax is not None, "If fig and ax are provided, both must be provided."
        if tube_colors is None:
            tube_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # generate colors, one for each tube, according to the default matplotlib color cycle
        for i, tube in enumerate(self.tubes):
            fig, ax = tube.plot_3d(fig=fig, ax=ax, save_path=None, xlim=xlim, ylim=ylim, zlim=zlim, expand_axes=expand_axes, plot_curve_shadows=plot_curve_shadows, plot_planes=(i == 0), tube_color=tube_colors[i], with_symmetry=with_symmetry)
        for connector in self.connectors:
            fig, ax = connector.plot(fig=fig, ax=ax, save_path=None, xlim=xlim, ylim=ylim, zlim=zlim, facecolor=connector_face_color, alpha=connector_alpha, show_arm_faces=show_arm_faces)
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig, ax
    
    def plot_unrolled_strips(self, save_path=None, xlim=None, ylim=None, extend_aabb=0.0, selected_strips=None, y_offset_per_tube=None, y_offset_per_strip=None, strip_colors=None, with_symmetry=False):
        fig, ax = plt.subplots()
        aabb_extent = torch.zeros(size=(2, 3))
        xlim_final = [np.inf, -np.inf]
        ylim_final = [np.inf, -np.inf]
        y_offset_total = 0.0
        strip_idx = 0
        for ti, tube in enumerate(self.tubes):
            y_offset_total = (1 + extend_aabb) * (aabb_extent[1, 1] - aabb_extent[0, 1])
            if y_offset_per_tube is not None:
                y_offset_total += y_offset_per_tube
            if y_offset_per_strip is not None:
                y_offset_total += y_offset_per_strip
            strip_indices = [i for i in range(strip_idx, strip_idx+tube.N)]
            if selected_strips is not None:
                strip_indices = [i for i in strip_indices if i in selected_strips]
            if strip_colors is not None:
                tube_strip_colors = [strip_colors[i] for i in strip_indices]
            else:
                tube_strip_colors = len(strip_indices) * [plt.rcParams['axes.prop_cycle'].by_key()['color'][ti]]  # generate colors, tube.N copies of the ti-th color for current tube ti, according to the default matplotlib color cycle
            fig, ax = tube.plot_unrolled_strips(fig=fig, ax=ax, save_path=None, xlim=xlim, ylim=ylim, selected_strips=selected_strips, y_offset_total=y_offset_total, y_offset_per_strip=y_offset_per_strip, strip_colors=tube_strip_colors, with_symmetry=with_symmetry)
            aabb_extent_curr = ax.get_xlim(), ax.get_ylim()
            aabb_extent_curr = torch.tensor([[aabb_extent_curr[0][0], aabb_extent_curr[1][0]], [aabb_extent_curr[0][1], aabb_extent_curr[1][1]]])
            aabb_extent[0, :2] = torch.min(aabb_extent[0, :2], aabb_extent_curr[0, :2])  # cumulate aabb
            aabb_extent[1, :2] = torch.max(aabb_extent[1, :2], aabb_extent_curr[1, :2])
            xlim_final = [min(xlim_final[0], aabb_extent[0, 0].item()), max(xlim_final[1], aabb_extent[1, 0].item())]
            ylim_final = [min(ylim_final[0], aabb_extent[0, 1].item()), max(ylim_final[1], aabb_extent[1, 1].item())]
            if y_offset_per_strip is not None:
                y_offset_total += y_offset_per_strip
            strip_idx += tube.N
            
        xrange = xlim_final[1] - xlim_final[0]
        yrange = ylim_final[1] - ylim_final[0]
        xlim_final = [xlim_final[0] - 0.05 * xrange, xlim_final[1] + 0.05 * xrange]
        ylim_final = [ylim_final[0] - 0.05 * yrange, ylim_final[1] + 0.05 * yrange]

        if xlim is not None:
            xlim_final = xlim
        if ylim is not None:
            ylim_final = ylim
        ax.set_xlim(xlim_final)
        ax.set_ylim(ylim_final)

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig, ax