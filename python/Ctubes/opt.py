from Ctubes.target_cross_sections import ConnectorTargetCrossSections
from Ctubes.tubes import CTube
from Ctubes.tube_networks import CTubeNetwork
from Ctubes.objectives import *
from Ctubes.geometry_utils import compute_curvature_angles
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from Ctubes.plot_utils import plot_tube, render_video, LINEWIDTH_REF
import torch
import igl
import dill
import os

TORCH_DTYPE = torch.float64


def obj_and_grad(params, opt_prob):
    params_torch = torch.tensor(params, dtype=TORCH_DTYPE)
    params_torch.requires_grad = True
    opt_prob.set_params(params_torch)
    obj = opt_prob.compute_objective()
    obj.backward()
    return obj.item(), params_torch.grad.numpy()


class CTubeOptimizationProblem:
    def __init__(self, tube, opt_weights, objective_args={}):
        '''
        Args:
            tube (CTube, CTubeNetwork): the tube(s) to optimize.
            opt_weights (dict): dictionary of the weights for the different objectives.
            objective_args (dict): dictionary of additional arguments to pass to the objective functions.
        '''
                
        if isinstance(tube, CTubeNetwork):
            self.tube_network = tube
        elif isinstance(tube, CTube):
            self.tube_network = CTubeNetwork(tubes=[tube], connectors=[], connector_to_tube_connectivity=[])
        else:
            raise ValueError("tube must be a either CTube or a CTubeNetwork, got {}.".format(type(tube)))

        self.objective_args = objective_args

        self.active_cps = False
        self.active_theta = False
        self.active_apex_loc_func = False
        self.active_plane_normals = False
        self.active_init_scale = False
        self.active_dihedral_angles = False
        self.active_connector_positions = False
        self.active_connector_orientations = False

        self.dof_index_offset_cps = 0
        self.dof_index_offset_theta = 0
        self.dof_index_offset_apex_loc_func = 0
        self.dof_index_offset_plane_normals = 0
        self.dof_index_offset_init_scale = 0
        self.dof_index_offset_dihedral_angles = 0
        self.dof_index_offset_connector_positions = 0
        self.dof_index_offset_connector_orientations = 0

        self.n_active_params = 0

        self.objective_current = {}
        self.objective_history = {}
        self.n_iters = 0
        self.opt_time = 0.0
        self.success = None
        self.message = None

        self.opt_weights = opt_weights
        self.update_default_weights()

    def update_n_active_params(self):
        self.n_active_params = \
            self.n_dofs_cps() + \
            self.n_dofs_theta() + \
            self.n_dofs_apex_loc_func() + \
            self.n_dofs_plane_normals() + \
            self.n_dofs_init_scale() + \
            self.n_dofs_dihedral_angles() + \
            self.n_dofs_connector_positions() + \
            self.n_dofs_connector_orientations()

    def n_dofs_cps(self):
        return sum([3 * tube.K for tube in self.tube_network.tubes]) - sum([3 * self.tube_network.n_closed_curves]) if self.active_cps else 0  # remove the last control point for closed curves
    
    def n_dofs_theta(self):
        return self.tube_network.n_tubes if self.active_theta else 0
    
    def n_dofs_apex_loc_func(self):
        return sum([tube.M - 1 for tube in self.tube_network.tubes]) if self.active_apex_loc_func else 0
    
    def n_dofs_plane_normals(self):
        if self.tube_network.has_planar_profile():
            return 3 * self.tube_network.M - 3 * self.tube_network.n_tubes_with_symmetry if self.active_plane_normals else 0  # remove the last normal for tubes with symmetry
        else:
            assert self.tube_network.n_tubes == 1, "Non-planar profiles are only supported for single tubes."
            return 3 * self.tube_network.tubes[0].M * self.tube_network.tubes[0].N if self.active_plane_normals else 0

    def n_dofs_init_scale(self):
        return self.tube_network.n_tubes if self.active_init_scale else 0
    
    def n_dofs_dihedral_angles(self):
        return sum([(tube.M-1) * tube.N for tube in self.tube_network.tubes]) if self.active_dihedral_angles else 0
    
    def n_dofs_connector_positions(self):
        return 3 * self.tube_network.n_connectors if self.active_connector_positions else 0
    
    def n_dofs_connector_orientations(self):
        return 3 * self.tube_network.n_connectors if self.active_connector_orientations else 0

    def update_dof_index_offsets(self):
        self.dof_index_offset_cps = 0
        self.dof_index_offset_theta = self.dof_index_offset_cps + self.n_dofs_cps()
        self.dof_index_offset_apex_loc_func = self.dof_index_offset_theta + self.n_dofs_theta()
        self.dof_index_offset_plane_normals = self.dof_index_offset_apex_loc_func + self.n_dofs_apex_loc_func()
        self.dof_index_offset_init_scale = self.dof_index_offset_plane_normals + self.n_dofs_plane_normals()
        self.dof_index_offset_dihedral_angles = self.dof_index_offset_init_scale + self.n_dofs_init_scale()
        self.dof_index_offset_connector_positions = self.dof_index_offset_dihedral_angles + self.n_dofs_dihedral_angles()
        self.dof_index_offset_connector_orientations = self.dof_index_offset_connector_positions + self.n_dofs_connector_positions()

    def activate_cps(self, active=True):
        self.active_cps = active
        if active and not self.active_plane_normals:
            for ti in range(self.tube_network.n_tubes):
                if self.tube_network.tubes[ti].plane_normal_kind is None:
                    print(f"Warning: Activating cps but plane_normal_kind was None for tube {ti}. Setting plane_normal_kind to 'bisecting' to ensure plane normals are updated during optimization.")
                    self.tube_network.tubes[ti].plane_normal_kind = 'bisecting'
                    self.tube_network.tubes[ti].update_plane_normals(kind='bisecting')
        self.update_n_active_params()
        self.update_dof_index_offsets()
        self.update_default_weights()

    def activate_theta(self, active=True):
        self.active_theta = active
        self.update_n_active_params()
        self.update_dof_index_offsets()
        self.update_default_weights()

    def activate_apex_loc_func(self, active=True):
        if active:
            assert not self.active_dihedral_angles, "Cannot activate apex_loc_func if dihedral_angles are active."
            for tube in self.tube_network.tubes:
                assert not tube.use_dihedral_angles(), "Cannot activate apex_loc_func if the tube is parametrized using dihedral angles."
        self.active_apex_loc_func = active
        self.update_n_active_params()
        self.update_dof_index_offsets()
        self.update_default_weights()

    def activate_plane_normals(self, active=True):
        if self.tube_network.has_planar_profile():
            for ti in range(self.tube_network.n_tubes):
                if self.tube_network.tubes[ti].plane_normal_kind is not None and active:
                    print(f"Warning: Activating plane normals in tube {ti} but plane_normal_kind was set to {self.tube_network.tubes[ti].plane_normal_kind}. Setting plane_normal_kind to None.")
                    self.tube_network.tubes[ti].plane_normal_kind = None
                elif self.tube_network.tubes[ti].plane_normal_kind is None and not active:
                    print(f"Warning: Deactivating plane normals in tube {ti} but plane_normal_kind was None. Leaving it to None and freezing existing plane normals. Set plane_normal_kind if you wish otherwise.")
        else:
            assert self.tube_network.n_tubes == 1, "Non-planar profiles are only supported for single tubes."
            if active:
                assert self.tube_network.tubes[0].plane_normal_kind is None, "Cannot activate plane_normals for non-planar profile if plane_normal_kind is set."
            else:
                print("WARNING: Deactivating plane_normals for non-planar profile. The normals will not be updated during the optimization. Are you sure?")
        self.active_plane_normals = active
        self.update_n_active_params()
        self.update_dof_index_offsets()
        self.update_default_weights()

    def activate_init_scale(self, active=True):
        self.active_init_scale = active
        self.update_n_active_params()
        self.update_dof_index_offsets()
        self.update_default_weights()

    def activate_dihedral_angles(self, active=True):
        if active:
            assert not self.active_apex_loc_func, "Cannot activate dihedral angles if apex_loc_func is active."
            for tube in self.tube_network.tubes:
                assert tube.use_dihedral_angles(), "Cannot activate dihedral angles if the tube is not parametrized using dihedral angles."
        if not active and any([tube.use_dihedral_angles() for tube in self.tube_network.tubes]):
            print("Warning: Deactivating dihedral angles but some tubes are parametrized using dihedral angles.\n Dihedral angles will not be updated during the optimization. Are you sure?")
        self.active_dihedral_angles = active
        self.update_n_active_params()
        self.update_dof_index_offsets()
        self.update_default_weights()
        
    def activate_connector_positions(self, active=True):
        self.active_connector_positions = active
        self.update_n_active_params()
        self.update_dof_index_offsets()
        self.update_default_weights()
        
    def activate_connector_orientations(self, active=True):
        self.active_connector_orientations = active
        self.update_n_active_params()
        self.update_dof_index_offsets()
        self.update_default_weights()

    def get_dof_indices_cps(self):
        if not self.active_cps:
            return []
        idx = self.dof_index_offset_cps
        didx = self.n_dofs_cps()
        return list(range(idx, idx+didx))

    def get_dof_indices_theta(self):
        if not self.active_theta:
            return []
        idx = self.dof_index_offset_theta
        didx = self.n_dofs_theta()
        return list(range(idx, idx+didx))
    
    def get_dof_indices_apex_loc_func(self):
        if not self.active_apex_loc_func:
            return []
        idx = self.dof_index_offset_apex_loc_func
        didx = self.n_dofs_apex_loc_func()
        return list(range(idx, idx+didx))
    
    def get_dof_indices_plane_normals(self):
        if not self.active_plane_normals:
            return []
        idx = self.dof_index_offset_plane_normals
        didx = self.n_dofs_plane_normals()
        return list(range(idx, idx+didx))
    
    def get_dof_indices_init_scale(self):
        if not self.active_init_scale:
            return []
        idx = self.dof_index_offset_init_scale
        didx = self.n_dofs_init_scale()
        return list(range(idx, idx+didx))
    
    def get_dof_indices_dihedral_angles(self):
        if not self.active_dihedral_angles:
            return []
        idx = self.dof_index_offset_dihedral_angles
        didx = self.n_dofs_dihedral_angles()
        return list(range(idx, idx+didx))
    
    def get_dof_indices_connector_positions(self):
        if not self.active_connector_positions:
            return []
        idx = self.dof_index_offset_connector_positions
        didx = self.n_dofs_connector_positions()
        return list(range(idx, idx+didx))
    
    def get_dof_indices_connector_orientations(self):
        if not self.active_connector_orientations:
            return []
        idx = self.dof_index_offset_connector_orientations
        didx = self.n_dofs_connector_orientations()
        return list(range(idx, idx+didx))

    def set_params(self, params):
        assert params.numel() == self.n_active_params, "Number of parameters {} does not match the number of active parameters {}.".format(params.numel(), self.n_active_params)
        idx = 0

        # Tubes
        if self.active_cps:
            didx = self.n_dofs_cps()
            if self.tube_network.n_closed_curves == 0:
                cps = params[idx:idx+didx].reshape(-1, 3)
            else:   # the last control point is not optimized in closed curves (it's just a copy of the first one)
                K = self.tube_network.K
                K_free = K - self.tube_network.n_closed_curves
                K_free_per_tube = [tube.K - 1 if tube.directrix.closed_curve else tube.K for tube in self.tube_network.tubes]
                assert sum(K_free_per_tube) == K_free
                assert didx == 3 * K_free
                cps_free = params[idx:idx+didx].reshape(K_free, 3)
                cps_free_per_tube = list(torch.split(cps_free, K_free_per_tube, dim=0))
                cps_per_tube = []
                for ti, tube in enumerate(self.tube_network.tubes):
                    if tube.directrix.closed_curve:
                        cps_per_tube.append(torch.cat([cps_free_per_tube[ti], cps_free_per_tube[ti][0].unsqueeze(0)], dim=0))  # duplicate the first control point
                    else:
                        cps_per_tube.append(cps_free_per_tube[ti])
                cps = torch.cat(cps_per_tube, dim=0)
                assert cps.shape[0] == K
            idx += didx
            self.tube_network.set_control_points(cps)
        if self.active_theta:
            didx = self.n_dofs_theta()
            theta = params[idx:idx+didx]
            idx += didx
            self.tube_network.set_thetas(theta)
        if self.active_apex_loc_func:
            didx = self.n_dofs_apex_loc_func()
            apex_loc_func = params[idx:idx+didx]
            idx += didx
            self.tube_network.set_apex_loc_func(apex_loc_func)
        if self.active_plane_normals:
            didx = self.n_dofs_plane_normals()
            if not self.tube_network.has_symmetry():
                if self.tube_network.has_planar_profile():
                    assert didx == 3 * self.tube_network.M
                    plane_normals = params[idx:idx+didx].reshape(-1, 3)
                else:
                    assert self.tube_network.n_tubes == 1, "Non-planar profiles are only supported for single tubes."
                    assert didx == 3 * self.tube_network.tubes[0].M * self.tube_network.tubes[0].N
                    # plane_normals = torch.swapaxes(params[idx:idx+didx].reshape(self.tube_network.tubes[0].N, self.tube_network.tubes[0].M, 3), 0, 1)
                    plane_normals = params[idx:idx+didx].reshape(self.tube_network.tubes[0].M, self.tube_network.tubes[0].N, 3)
                # plane_normals = params[idx:idx+didx].reshape(-1, 3)
            else:
                n_pn = self.tube_network.M
                n_pn_free = n_pn - self.tube_network.n_tubes_with_symmetry
                n_pn_free_per_tube = [tube.M - 1 if tube.has_symmetry() else tube.M for tube in self.tube_network.tubes]
                assert sum(n_pn_free_per_tube) == n_pn_free
                assert didx == 3 * n_pn_free
                plane_normals_free = params[idx:idx+didx].reshape(n_pn_free, 3)
                plane_normals_free_per_tube = list(torch.split(plane_normals_free, n_pn_free_per_tube, dim=0))
                plane_normals_per_tube = []
                for ti, tube in enumerate(self.tube_network.tubes):
                    if tube.has_symmetry():
                        pn_last = tube.symmetry_transforms[0](plane_normals_free_per_tube[ti][0].unsqueeze(0))
                        plane_normals_per_tube.append(torch.cat([plane_normals_free_per_tube[ti], pn_last], dim=0))
                    else:
                        plane_normals_per_tube.append(plane_normals_free_per_tube[ti])
                plane_normals = torch.cat(plane_normals_per_tube, dim=0)
                assert plane_normals.shape[0] == n_pn
            idx += didx
            self.tube_network.set_plane_normals(plane_normals)
        if self.active_init_scale:
            didx = self.n_dofs_init_scale()
            init_scale = params[idx:idx+didx]
            idx += didx
            self.tube_network.set_init_scales(init_scale)
        if self.active_dihedral_angles:
            didx = self.n_dofs_dihedral_angles()
            dihedral_angles = params[idx:idx+didx]
            dihedral_angles_per_tube_flat = list(torch.split(dihedral_angles, [(tube.M-1) * tube.N for tube in self.tube_network.tubes], dim=0))
            dihedral_angles_per_tube = [da.reshape(tube.M-1, tube.N) for da, tube in zip(dihedral_angles_per_tube_flat, self.tube_network.tubes)]
            idx += didx
            self.tube_network.set_dihedral_angles(dihedral_angles_per_tube)
        self.tube_network.update_ctube_vertices_per_tube()

        # Connectors
        if self.active_connector_positions:
            didx = self.n_dofs_connector_positions()
            connector_positions = params[idx:idx+didx].reshape(-1, 3)
            idx += didx
            self.tube_network.set_connector_positions(connector_positions)
        if self.active_connector_orientations:
            didx = self.n_dofs_connector_orientations()
            connector_orientations = params[idx:idx+didx].reshape(-1, 3)
            idx += didx
            self.tube_network.set_connector_orientations(connector_orientations)

    def get_params(self):
        params_list = []
        if self.active_cps:
            if self.tube_network.n_closed_curves == 0:
                cps = self.tube_network.get_control_points()
            else:   # the last control point is not optimized in closed curves (it's just a copy of the first one)
                cps = self.tube_network.get_control_points()
                K_per_tube = [tube.K for tube in self.tube_network.tubes]
                idx = 0
                cps_per_tube = []
                for ti, tube in enumerate(self.tube_network.tubes):
                    if tube.directrix.closed_curve:
                        cps_per_tube.append(cps[idx:idx+K_per_tube[ti]-1])
                    else:
                        cps_per_tube.append(cps[idx:idx+K_per_tube[ti]])
                    idx += K_per_tube[ti]
                cps = torch.cat(cps_per_tube, dim=0)
            params_list.append(cps.flatten())
        if self.active_theta:
            params_list.append(self.tube_network.get_thetas())
        if self.active_apex_loc_func:
            params_list.append(self.tube_network.get_apex_loc_func())
        if self.active_plane_normals:
            if not self.tube_network.has_symmetry():
                plane_normals = self.tube_network.get_plane_normals()
            else:  # the last normal is not optimized in tubes with symmetry
                plane_normals = self.tube_network.get_plane_normals()
                n_pn_per_tube = [tube.M for tube in self.tube_network.tubes]
                idx = 0
                plane_normals_per_tube = []
                for ti, tube in enumerate(self.tube_network.tubes):
                    if tube.has_symmetry():
                        plane_normals_per_tube.append(plane_normals[idx:idx+n_pn_per_tube[ti]-1])
                    else:
                        plane_normals_per_tube.append(plane_normals[idx:idx+n_pn_per_tube[ti]])
                    idx += n_pn_per_tube[ti]
                plane_normals = torch.cat(plane_normals_per_tube, dim=0)
            params_list.append(plane_normals.flatten())
        if self.active_init_scale:
            params_list.append(self.tube_network.get_init_scales())
        if self.active_dihedral_angles:
            list_of_flattened_dihedral_angles = [tube.get_dihedral_angles().flatten() for tube in self.tube_network.tubes]
            params_list.append(torch.cat(list_of_flattened_dihedral_angles, dim=0))
        if self.active_connector_positions:
            params_list.append(self.tube_network.get_connector_positions().flatten())
        if self.active_connector_orientations:
            params_list.append(self.tube_network.get_connector_orientations().flatten())

        params = torch.cat(params_list, dim=0) if params_list else torch.tensor([], dtype=TORCH_DTYPE)
        return params

    def get_params_numpy(self):
        return self.get_params().detach().numpy()

    def set_opt_weights(self, opt_weights):
        self.opt_weights = opt_weights
        self.update_default_weights()

    def update_default_weights(self):
        '''Add the weights of the default objective terms that are always active.'''

        # Always active
        if 'smooth_plane_normal_diffs' not in self.opt_weights:
            self.opt_weights['smooth_plane_normal_diffs'] = 1e0

        if 'preserve_tube_ridge_edge_directions' not in self.opt_weights:
            self.opt_weights['preserve_tube_ridge_edge_directions'] = 1e2 / self.tube_network.tubes[0].directrix.aabb_diagonal_length() ** 2

        # Plane normals
        if self.active_plane_normals:
            if 'unitary_plane_normals' not in self.opt_weights:
                self.opt_weights['unitary_plane_normals'] = 1e-5

        # Apex location function
        if self.active_apex_loc_func:
            if 'smooth_apex_loc_func' not in self.opt_weights:
                self.opt_weights['smooth_apex_loc_func'] = 1e1


    def set_objective_args(self, objective_args):
        self.objective_args = objective_args
        
    def clean_gradients(self):
        params0 = self.get_params_numpy()
        params_torch = torch.tensor(params0, dtype=TORCH_DTYPE)
        self.set_params(params_torch)
        _ = self.compute_objective()

    def compute_objective(self, print_to_console=False):
        self.objective_current = {}
        eval_obj_terms = {}
        for key in self.opt_weights.keys():
            self.objective_current[key] = torch.tensor(0.0)
            eval_obj_terms[key] = False

        # ---------- Directrix ----------

        if 'preserve_curve' in self.opt_weights:
            if 'preserve_curve_mask' in self.objective_args:
                for ti, tube in enumerate(self.tube_network.tubes):
                    directrix_masked = tube.directrix.X[self.objective_args['preserve_curve_mask'][ti]]
                    directrix_ref_masked = tube.directrix_ref.X[self.objective_args['preserve_curve_mask'][ti]]
                    self.objective_current['preserve_curve'] = self.opt_weights['preserve_curve'] * preserve_curve(directrix_masked, directrix_ref_masked)
            else:  # no mask (all points are active)
                for tube in self.tube_network.tubes:
                    self.objective_current['preserve_curve'] += self.opt_weights['preserve_curve'] * preserve_curve(tube.directrix.X, tube.directrix_ref.X)
            eval_obj_terms['preserve_curve'] = True

        if 'min_directrix_self_distance' in self.opt_weights:
            assert self.tube_network.n_tubes == 1, "min_directrix_self_distance is only supported for a single tube."
            tube = self.tube_network.tubes[0]
            self.objective_current['min_directrix_self_distance'] = self.opt_weights['min_directrix_self_distance'] * min_directrix_self_distance(tube.get_polyline(), target_min_dist=self.objective_args['min_directrix_self_target_distance'] , closed_curve=tube.directrix.closed_curve)
            eval_obj_terms['min_directrix_self_distance'] = True

        if 'minimize_curvature' in self.opt_weights:
            for tube in self.tube_network.tubes:
                first_last_overlap = torch.le(torch.linalg.norm(tube.directrix.X[0] - tube.directrix.X[-1]), torch.tensor(1.0e-5))
                curvature_angles = compute_curvature_angles(tube.directrix.X, tube.directrix.closed_curve, first_last_overlap)
                self.objective_current['minimize_curvature'] += self.opt_weights['minimize_curvature'] * torch.mean(curvature_angles ** 2) / len(self.tube_network.tubes)

        # ---------- Cross-Section ----------

        if 'join_ends' in self.opt_weights:
            for tube in self.tube_network.tubes:
                if tube.directrix.closed_curve or (not tube.directrix.closed_curve and tube.directrix.first_last_points_match()):
                    self.objective_current['join_ends'] += self.opt_weights['join_ends'] * join_ends(tube.ctube_vertices.reshape(-1, 3), tube.ctube_vertices.shape[1], self.objective_args['join_ends_pairings'])
                if not tube.directrix.closed_curve and tube.has_symmetry():
                    # Check that the last point of the transformed base curve matches the first point of the original base curve
                    directrix = tube.get_polyline_with_symmetry()
                    assert torch.allclose(directrix[-1], directrix[0]), "The last point of the transformed base curve {} does not match the first point of the original base curve {}.".format(directrix[-1], directrix[0])
                    ctube_vertices = tube.compute_vertices_with_symmetry()
                    self.objective_current['join_ends'] += self.opt_weights['join_ends'] * join_ends(ctube_vertices.reshape(-1, 3), ctube_vertices.shape[1], self.objective_args['join_ends_pairings'])
            eval_obj_terms['join_ends'] = True

        if 'match_target_cross_sections' in self.opt_weights:
            for ti, tube in enumerate(self.tube_network.tubes):
                tcs = self.objective_args['target_cross_sections']
                target_points = tcs.get_target_points_for_tube(ti)
                cross_section_indices = tcs.get_cross_section_indices_for_tube(ti)
                pairings = tcs.get_pairings_for_tube(ti)
                self.objective_current['match_target_cross_sections'] += self.opt_weights['match_target_cross_sections'] * match_target_cross_sections(tube.ctube_vertices, cross_section_indices, target_points, pairings, project_on_target_plane=True)
            eval_obj_terms['match_target_cross_sections'] = True

        if 'match_target_cross_section_radius' in self.opt_weights and 'target_cross_section_radii' in self.objective_args:
            for ti, tube in enumerate(self.tube_network.tubes):
                self.objective_current['match_target_cross_section_radius'] += self.opt_weights['match_target_cross_section_radius'] * match_target_cross_section_radius(tube.directrix.X, tube.ctube_vertices, self.objective_args['target_cross_section_radii'])
            eval_obj_terms['match_target_cross_section_radius'] = True

        if 'match_connector_target_cross_sections' in self.opt_weights:
            for ti, tube in enumerate(self.tube_network.tubes):
                target_points = self.connector_target_cross_sections.get_target_points_for_tube(ti)
                cross_section_indices = self.connector_target_cross_sections.get_cross_section_indices_for_tube(ti)
                pairings = self.connector_target_cross_sections.get_pairings_for_tube(ti)
                self.objective_current['match_connector_target_cross_sections'] += self.opt_weights['match_connector_target_cross_sections'] * match_target_cross_sections(tube.ctube_vertices, cross_section_indices, target_points, pairings)
            eval_obj_terms['match_connector_target_cross_sections'] = True

        # ---------- Alignment ----------

        if 'constrain_tube_ridges_to_surface' in self.opt_weights:
            for ti in range(self.tube_network.n_tubes):
                self.objective_current['constrain_tube_ridges_to_surface'] += self.opt_weights['constrain_tube_ridges_to_surface'] * constrain_tube_ridges_to_surface(self.tube_network.tubes[ti].ctube_vertices, self.objective_args['target_surface']['vertices'], self.objective_args['target_surface']['faces'], self.objective_args['constrained_tube_ridges'])
            eval_obj_terms['constrain_tube_ridges_to_surface'] = True

        if 'quad_tangency' in self.opt_weights:
            self.objective_current['quad_tangency'] = self.opt_weights['quad_tangency'] * quad_tangency(self.tube_network.compute_vertices(), self.objective_args['quad_pair_tube_indices'], self.objective_args['quad_pair_disc_indices'], self.objective_args['quad_pair_cross_section_indices'])
            eval_obj_terms['quad_tangency'] = True

        if 'quad_orientation' in self.opt_weights:
            self.objective_current['quad_orientation'] = self.opt_weights['quad_orientation'] * quad_orientation(self.tube_network.compute_vertices(), self.objective_args['quad_orient_tube_indices'], self.objective_args['quad_orient_disc_indices'], self.objective_args['quad_orient_cross_section_indices'], self.objective_args['quad_orient_target'])
            eval_obj_terms['quad_orientation'] = True

        if 'quad_tangency_any_pairing' in self.opt_weights:
            self.objective_current['quad_tangency_any_pairing'] = self.opt_weights['quad_tangency_any_pairing'] * quad_tangency_any_pairing(self.tube_network.compute_vertices(), self.tube_network.get_polylines(), self.objective_args['quad_pair_tube_indices_any_pairing'], self.objective_args['quad_pair_disc_indices_any_pairing'])
            eval_obj_terms['quad_tangency_any_pairing'] = True

        if 'quad_distance' in self.opt_weights:
            self.objective_current['quad_distance'] = self.opt_weights['quad_distance'] * quad_distance(self.tube_network.compute_vertices(), self.objective_args['quad_face_dist_tube_indices'], self.objective_args['quad_face_dist_disc_indices'], self.objective_args['quad_face_dist_cross_section_indices'], self.objective_args['quad_face_dist_target_distances'])
            eval_obj_terms['quad_distance'] = True

        # ---------- Regularization ----------

        if 'smooth_plane_normal_diffs' in self.opt_weights:
            for tube in self.tube_network.tubes:
                if self.tube_network.has_planar_profile():
                    self.objective_current['smooth_plane_normal_diffs'] += self.opt_weights['smooth_plane_normal_diffs'] * smooth_plane_normal_diffs(tube.plane_normals, tube.directrix.closed_curve)
                else:
                    for cs in range(tube.N):
                        self.objective_current['smooth_plane_normal_diffs'] += self.opt_weights['smooth_plane_normal_diffs'] * smooth_plane_normal_diffs(tube.plane_normals[:, cs], tube.directrix.closed_curve)
            eval_obj_terms['smooth_plane_normal_diffs'] = True

        if 'smooth_apex_loc_func' in self.opt_weights:
            for tube in self.tube_network.tubes:
                self.objective_current['smooth_apex_loc_func'] += self.opt_weights['smooth_apex_loc_func'] * smooth_apex_loc_func(tube.get_apex_loc_func(), closed_curve=tube.directrix.closed_curve)
            eval_obj_terms['smooth_apex_loc_func'] = True

        if 'smooth_tube_ridges' in self.opt_weights:
            for tube in self.tube_network.tubes:
                self.objective_current['smooth_tube_ridges'] += self.opt_weights['smooth_tube_ridges'] * smooth_tube_ridges(tube.ctube_vertices, tube.directrix.closed_curve)
            eval_obj_terms['smooth_tube_ridges'] = True

        if 'preserve_tube_ridge_edge_directions' in self.opt_weights:
            for tube in self.tube_network.tubes:
                min_length = 1e-1 * tube.reference_length() / (tube.M - 1)
                self.objective_current['preserve_tube_ridge_edge_directions'] += self.opt_weights['preserve_tube_ridge_edge_directions'] * preserve_tube_ridge_edge_directions(tube.ctube_vertices, tube.directrix.X, min_length)
            eval_obj_terms['preserve_tube_ridge_edge_directions'] = True

        if 'unitary_plane_normals' in self.opt_weights:
            for tube in self.tube_network.tubes:
                self.objective_current['unitary_plane_normals'] += self.opt_weights['unitary_plane_normals'] * unit_vectors(tube.plane_normals)
            eval_obj_terms['unitary_plane_normals'] = True

        # Check that all the objectives required by the weights were computed
        for key in self.opt_weights.keys():
            if key not in eval_obj_terms or not eval_obj_terms[key]:
                raise ValueError(f"Objective term '{key}' was not evaluated. Please check the weights and the objective arguments.")  

        obj = sum([v for k, v in self.objective_current.items()])

        if print_to_console:
            for k, v in self.objective_current.items():
                print(f"{k:40}: {v.item():.10f}")

        return obj
    
    def add_objective_to_history(self):
        for k, v in self.objective_current.items():
            if k not in self.objective_history and self.n_iters == 0:
                self.objective_history[k] = []
            elif k not in self.objective_history and self.n_iters > 0:
                self.objective_history[k] = self.n_iters * [0.0]
            else:
                self.objective_history[k].append(v.item())
        for k, v in self.objective_history.items():
            if k not in self.objective_current:
                self.objective_history[k].append(0.0)  # the objective term was deactivated: keep populating the corresponding entry with zeros
        self.n_iters += 1

    def save_json(self, file_path):
        import json

        # Optimization data
        def item_to_list(v):
            if isinstance(v, torch.Tensor):
                return v.detach().numpy().tolist()
            elif isinstance(v, list):
                return [item_to_list(x) for x in v]
            elif isinstance(v, dict):
                return {k: item_to_list(val) for k, val in v.items()}
            elif hasattr(v, 'to_dict') and callable(getattr(v, 'to_dict')):
                return v.to_dict()
            else:
                return v
        def to_list(d):
            return {k: item_to_list(v) for k, v in d.items()}
            
        opt_dict = {
            'opt_weights': to_list(self.opt_weights),
            'objective_args': to_list(self.objective_args),
            'opt_history': self.objective_history,
            'opt_time': self.opt_time,
            'n_iters': self.n_iters,
            'success': self.success,
            'message': self.message,
            'active_cps': self.active_cps,
            'active_theta': self.active_theta,
            'active_apex_loc_func': self.active_apex_loc_func,
            'active_plane_normals': self.active_plane_normals,
            'active_init_scale': self.active_init_scale,
            'active_dihedral_angles': self.active_dihedral_angles,
            'active_connector_positions': self.active_connector_positions,
            'active_connector_orientations': self.active_connector_orientations,
            'params_opt': self.get_params().tolist(),
        }

        # Geometry data, per tube
        geo_dict = self.tube_network.get_geometry()

        # Save to JSON
        with open(file_path, 'w') as f:
            json.dump({'optimization': opt_dict, 'geometry': geo_dict}, f, indent=4)

    def plot(self, save_path=None, xlim=None, ylim=None):
        fig, ax = plt.subplots()
        for ti, tube in enumerate(self.tube_network.tubes):
            tcs_points = self.objective_args['target_cross_sections'].get_target_points_for_tube(ti) if 'target_cross_sections' in self.objective_args else None
            fig, ax = plot_tube(tube.directrix.X, tube.ctube_vertices, fig=fig, ax=ax, save_path=None, xlim=xlim, ylim=ylim, target_cross_sections=tcs_points)
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig, ax
    
    def plot_3d(self, save_path=None, xlim=None, ylim=None, zlim=None, tube_colors=None, target_cross_section_color='r', with_symmetry=True):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        fig, ax = self.tube_network.plot_3d(fig=fig, ax=ax, save_path=None, xlim=xlim, ylim=ylim, zlim=zlim, tube_colors=tube_colors, with_symmetry=with_symmetry)

        if 'target_surface' in self.objective_args:
            import matplotlib.tri as mtri
            V = self.objective_args['target_surface']['vertices']
            F = self.objective_args['target_surface']['faces']

            # Plot surface
            triangles = mtri.Triangulation(V[:,0], V[:,1], triangles=F)
            ax.plot_trisurf(triangles, V[:, 2], linewidth=0, antialiased=True, alpha=0.5, color='gray')

            # Plot boundary
            bnd = igl.boundary_loop(F.numpy())
            ax.plot(V[bnd, 0], V[bnd, 1], V[bnd, 2], 'k--', linewidth=0.5)

        # Quad tangency
        if 'quad_pair_tube_indices' in self.objective_args and 'quad_pair_disc_indices' in self.objective_args and 'quad_pair_cross_section_indices' in self.objective_args and 'quad_tangency' in self.opt_weights:
            quad_pair_tube_indices_flat = self.objective_args['quad_pair_tube_indices'].flatten()
            quad_pair_disc_indices_flat = self.objective_args['quad_pair_disc_indices'].flatten()
            quad_pair_cross_section_indices_flat = self.objective_args['quad_pair_cross_section_indices'].flatten()
            vertices = self.tube_network.compute_vertices()
            normals = compute_quad_normals(vertices, quad_pair_tube_indices_flat, quad_pair_disc_indices_flat, quad_pair_cross_section_indices_flat)
            for i in range(normals.shape[0]):
                ti = quad_pair_tube_indices_flat[i]
                di = quad_pair_disc_indices_flat[i]
                ci = quad_pair_cross_section_indices_flat[i]
                cip1 = (ci + 1) % vertices[ti].shape[1]
                pt = (vertices[ti][di, ci] + vertices[ti][di, cip1] + vertices[ti][di+1, cip1] + vertices[ti][di+1, ci]) / 4
                pt = pt.detach().numpy()
                n = normals[i].detach().numpy()
                color = tube_colors[ti] if tube_colors is not None else 'k'
                ax.quiver(pt[0], pt[1], pt[2], n[0], n[1], n[2], color=color, length=1, normalize=False)

        # Quad orientation
        if 'quad_orient_tube_indices' in self.objective_args and 'quad_orient_disc_indices' in self.objective_args and 'quad_orient_cross_section_indices' in self.objective_args and 'quad_orient_target' in self.objective_args and 'quad_orientation' in self.opt_weights:
            quad_orient_tube_indices = self.objective_args['quad_orient_tube_indices']
            quad_orient_disc_indices = self.objective_args['quad_orient_disc_indices']
            quad_orient_cross_section_indices = self.objective_args['quad_orient_cross_section_indices']
            quad_orient_target = self.objective_args['quad_orient_target']
            vertices = self.tube_network.compute_vertices()
            normals = compute_quad_normals(vertices, quad_orient_tube_indices, quad_orient_disc_indices, quad_orient_cross_section_indices)
            for i in range(len(quad_orient_tube_indices)):
                ti = quad_orient_tube_indices[i]
                di = quad_orient_disc_indices[i]
                ci = quad_orient_cross_section_indices[i]
                cip1 = (ci + 1) % vertices[ti].shape[1]
                pt = (vertices[ti][di, ci] + vertices[ti][di, cip1] + vertices[ti][di+1, cip1] + vertices[ti][di+1, ci]) / 4
                pt = pt.detach().numpy()
                target = quad_orient_target[i]
                n = normals[i].detach().numpy()
                color = tube_colors[ti] if tube_colors is not None else 'k'
                ax.quiver(pt[0], pt[1], pt[2], n[0], n[1], n[2], color=color, length=1, normalize=False)
                ax.quiver(pt[0], pt[1], pt[2], target[0], target[1], target[2], color=color, length=1, normalize=False, alpha=0.3)

        # Target cross sections
        verts = []
        if 'target_cross_sections' in self.objective_args:
            verts += [pts.detach().numpy() for pts in self.objective_args['target_cross_sections'].get_target_points()]
        if self.tube_network.n_connectors > 0:
            verts += list(self.connector_target_cross_sections.get_target_points().detach().numpy()) # add a list of arrays of shape (n_points_cs, 3)
        poly = Poly3DCollection(verts, color=target_cross_section_color, alpha=0.5, linewidth=LINEWIDTH_REF)
        ax.add_collection3d(poly)

        # Preserved points
        if 'preserve_curve_mask' in self.objective_args and 'preserve_curve' in self.opt_weights:
            for ti, tube in enumerate(self.tube_network.tubes):
                directrix_masked = tube.directrix.X[self.objective_args['preserve_curve_mask'][ti]].detach().numpy()
                ax.scatter(directrix_masked[:, 0], directrix_masked[:, 1], directrix_masked[:, 2], c=target_cross_section_color, s=20, marker='o')
        
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig, ax

    def plot_unrolled_strips(self, save_path=None, xlim=None, ylim=None, extend_aabb=0.0, selected_strips=None, y_offset_per_tube=None, y_offset_per_strip=None, with_symmetry=True):
        return self.tube_network.plot_unrolled_strips(save_path=save_path, xlim=xlim, ylim=ylim, extend_aabb=extend_aabb, selected_strips=selected_strips, y_offset_per_tube=y_offset_per_tube, y_offset_per_strip=y_offset_per_strip, with_symmetry=with_symmetry)

    def plot_objective_history(self, save_path=None):
        fig, ax = plt.subplots()
        for k, v in self.objective_history.items():
            v_plot = np.array(v)
            v_plot[np.abs(v_plot) < 1e-15] = np.nan  # skip zero values (log scale)
            ax.plot(v_plot, label=k)
        ax.legend()
        ax.set_yscale('log')
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig, ax

    def plot_cross_section_radii(self):
        fig, ax = plt.subplots()
        for ti, tube in enumerate(self.tube_network.tubes):
            color = 'C{}'.format(ti)
            curr_cs_radii = compute_cross_section_radii(tube.get_polyline(), tube.compute_vertices()).detach().numpy()
            if 'target_cross_section_radii' in self.objective_args:
                assert self.tube_network.n_tubes == 1, "Plotting target cross-section radii only supported for a single tube."
                target_cs_radii = self.objective_args['target_cross_section_radii'].detach().numpy()
                ax.plot(target_cs_radii, marker='x', label='Target cross-section radii', color=color, linestyle='--')
            ax.plot(curr_cs_radii, marker='.', label='Current cross-section radii', color=color, linestyle='-')
        ax.set_xlabel('Cross-section index')
        ax.set_ylabel('Radius')
        ax.set_title('Cross-section radii')
        ax.legend()
        return fig, ax
    
    def plot_apex_loc_func(self):
        fig, ax = plt.subplots()
        for ti, tube in enumerate(self.tube_network.tubes):
            color = 'C{}'.format(ti)
            apex_loc_func = tube.get_apex_loc_func().detach().numpy()
            ax.plot(apex_loc_func, marker='.', color=color, linestyle='-')
        ax.set_xlabel('Cross-section index')
        ax.set_ylabel('Apex-locating function value')
        ax.set_title('Apex-locating function')
        return fig, ax

    # ================= Optimization output configuration and callback =================

    def configure_optimization_output(
            self, output_paths, record_movie=False, record_unrolled_strips=False, 
            save_iter_to_obj=False, save_iter_to_json=False, save_directrix=False, 
            render_iter=15, fps=15, offscreen=True
        ):
        """
        Configure output options for optimization.
        """
        self.output_paths = output_paths
        self.record_movie = record_movie
        self.record_unrolled_strips = record_unrolled_strips
        self.save_iter_to_obj = save_iter_to_obj
        self.save_iter_to_json = save_iter_to_json
        self.save_directrix = save_directrix
        self.render_iter = render_iter
        self.fps = fps
        self.offscreen = offscreen
        self.opt_iter = 0
        
        # Set up plotting limits if needed
        if hasattr(self, 'tube_network'):
            if self.tube_network.n_tubes == 1:
                # Single tube case
                ctube_vertices = self.tube_network.tubes[0].compute_vertices_with_symmetry()
                self.xlim, self.ylim, self.zlim = None, None, None  # Let plot methods handle this
                
                # For unrolled strips
                from Ctubes.tube_generation import compute_unrolled_strips
                vertices_per_strip, _ = compute_unrolled_strips(ctube_vertices)
                unrolled_strips = torch.cat(vertices_per_strip, dim=0)
                self.xlim_strip, self.ylim_strip, _ = None, None, None  # Let plot methods handle this
            else:
                # Multiple tubes case
                directrix_ref_stacked = torch.cat(self.tube_network.get_polylines_with_symmetry(), dim=0)
                self.xlim, self.ylim, self.zlim = None, None, None  # Let plot methods handle this
                self.xlim_strip, self.ylim_strip = None, None
        
        if self.offscreen:
            plt.ioff()  # deactivate interactive mode

    def optimization_callback(self, params, tube_colors=None, target_cross_section_color='r', with_symmetry=True):
        """
        Callback function to be used during optimization for saving and rendering.
        """
        self.opt_iter += 1
        
        # Update parameters
        params_torch = torch.tensor(params, dtype=TORCH_DTYPE)
        self.set_params(params_torch)
        
        # Render and save if needed
        if self.record_movie and self.opt_iter % self.render_iter == 0:
            save_path = os.path.join(self.output_paths["output"], f"tube_{self.opt_iter:04d}.png")
            self.plot_3d(save_path, xlim=self.xlim, ylim=self.ylim, zlim=self.zlim, tube_colors=tube_colors, target_cross_section_color=target_cross_section_color, with_symmetry=with_symmetry)

        if self.record_unrolled_strips and self.opt_iter % self.render_iter == 0:
            save_path = os.path.join(self.output_paths["output"], f"unrolled_strips_{self.opt_iter:04d}.png")
            self.plot_unrolled_strips(save_path, xlim=self.xlim_strip, ylim=self.ylim_strip, with_symmetry=with_symmetry)

        if self.save_iter_to_obj and self.opt_iter % self.render_iter == 0:
            save_path = os.path.join(self.output_paths["output_meshes"], f"tube_{self.opt_iter:04d}.obj")
            self.tube_network.save_obj(save_path, save_directrix=self.save_directrix)

        if self.save_iter_to_json and self.opt_iter % self.render_iter == 0:
            save_path = os.path.join(self.output_paths["output_meshes"], f"tube_json_{self.opt_iter:04d}.json")
            self.tube_network.save_json(save_path)
        
        # Always add to objective history
        self.add_objective_to_history()

    def save_optimization_results(self, paths=None):
        """
        Save final optimization results (pkl and json files).
        """
        paths = self.output_paths if paths is None else paths

        # Save as pickle
        pkl_file = os.path.join(paths["output_opt"], "opt_prob.pkl")
        dill.dump(self, open(pkl_file, "wb"))
        
        # Save as JSON
        json_file = os.path.join(paths["output_opt"], "opt_prob.json")
        self.save_json(json_file)

    def save_meshes(self, paths=None):
        """
        Save final meshes (OBJ files).
        """
        assert len(self.tube_network.tubes) == 1, "save_meshes currently assumes a single tube."
        tube = self.tube_network.tubes[0]

        paths = self.output_paths if paths is None else paths
        name = paths["name"]

        save_path = os.path.join(paths["output_meshes"], "{}_3d_tube.obj".format(name))
        self.tube_network.save_obj(save_path, split_strips=False)

        save_path = os.path.join(paths["output_meshes"], "{}_3d_tube_split_strips.obj".format(name))
        tube.save_obj(save_path, split_strips=True)

        save_path = os.path.join(paths["output_meshes"], "{}_2d_strips.obj".format(name))
        tube.save_strips_obj(save_path)

    def render_videos(self):
        """
        Render final videos from saved frames.
        """
        if self.record_movie:
            render_video("tube_", "tube.mp4", self.output_paths["output"], fps=self.fps)
        
        if self.record_unrolled_strips:
            render_video("unrolled_strips_", "unrolled_strips.mp4", self.output_paths["output"], fps=self.fps)

    def finalize_optimization(self, result):
        """
        Finalize optimization: save results, render videos, restore plotting state.
        
        Args:
            result: Result from scipy.optimize.minimize
        """
        self.opt_time += result.execution_time if hasattr(result, 'execution_time') else 0.0
        self.success = result.success
        self.message = result.message
        
        # Save results
        self.save_optimization_results()
        self.save_meshes()
        
        # Render videos
        self.render_videos()
        
        # Restore plotting state
        if self.offscreen:
            plt.close('all')
            plt.ion()  # re-enable interactive mode