from Ctubes.connectors import Connector
from Ctubes.tubes import CTube
import torch
from Ctubes.misc_utils import get_pairings_exact, get_pairings_all

# Class for target cross-sections.
# The class stores the target cross-sections as well as the indices of the cross-sections to be matched to the tube.
class TargetCrossSections:
    "Serves for both CTube and CTubeNetwork. For a single tube, the tube_indices will be [0]."
    def __init__(self, tube_indices, cross_section_indices, pairings):
        '''
        Args:
            tube_indices (torch.tensor of shape (n_target_cross_sections,)): indices of tubes to use for each cross section.
            cross_section_indices (torch.tensor of shape (n_target_cross_sections,)): indices of cross section indices to use for each tube.
            pairings (torch.tensor of shape (n_target_cross_sections, n_pairings, n_points_cs, 2)): tensor of the allowed pairings between the indices of the target cross section [..., 0] and the candidate one in the tube [..., 1]. Pick n_pairings as the maximum amount of pairing needed and repeat pairings if necessary.
        '''
        self.tube_indices = tube_indices
        self.tube_idx_to_local_indices_index = {tube_idx.item(): [] for tube_idx in tube_indices}
        for i, tube_idx in enumerate(tube_indices):
            self.tube_idx_to_local_indices_index[tube_idx.item()].append(i)
        self.n_target_cross_sections = tube_indices.shape[0]
        self.cross_section_indices = cross_section_indices
        self.pairings = pairings
        self.target_cross_sections_id_for_tubes = None # dictionary of the target cross-section indices for each tube index
        self.cross_section_indices_for_tubes = None # dictionary of the cross-section indices to use along the tube for each tube index
        self.compute_indices_for_tubes()

    def compute_indices_for_tubes(self):
        self.target_cross_sections_id_for_tubes = {tube_idx.item(): torch.tensor([i for i, x in enumerate(self.tube_indices) if x == tube_idx], dtype=torch.int32) for tube_idx in self.tube_indices}
        self.cross_section_indices_for_tubes = {tube_idx.item(): torch.tensor([self.cross_section_indices[i.item()] for i in self.target_cross_sections_id_for_tubes[tube_idx.item()]], dtype=torch.int32) for tube_idx in self.tube_indices}

    def set_tube_indices(self, tube_indices):
        self.tube_indices = tube_indices
        self.compute_indices_for_tubes()

    def set_cross_section_indices(self, cross_section_indices):
        self.cross_section_indices = cross_section_indices
        self.compute_indices_for_tubes()

    def set_pairings(self, pairings):
        self.pairings = pairings
    
    def get_cross_section_indices_for_tube(self, tube_idx):
        "Get the cross-section indices for the tube with index tube_idx."
        assert tube_idx in self.tube_indices, "The tube index must be in the tube_indices."
        # Find the positions of all the occurrences of tube_idx in the tube_indices
        return self.cross_section_indices_for_tubes[tube_idx]
    
    def get_pairings_for_tube(self, tube_idx):
        "Get the pairings for the tube with index tube_idx."
        assert tube_idx in self.tube_indices, "The tube index must be in the tube_indices."
        return self.pairings[tube_idx]
    
    def get_target_points_for_tube(self, tube_idx):
        '''
        Args:
            tube_idx (int): index of the tube.
            
        Returns:
            target_points (torch.Tensor of shape (Ns, n_points_cs, 3)): tensor of the target points for the indicated tube.
        '''
        raise NotImplementedError("This method should be implemented in the child class.")
    
    def get_target_points(self):
        '''
        Returns:
            target_points (torch.Tensor of shape (Ns, n_points_cs, 3)): tensor of the target points.
        '''
        return torch.cat([self.get_target_points_for_tube(tube_idx) for tube_idx in self.tube_indices], dim=0)

    def to_dict(self):
        """Convert to JSON-serializable dictionary."""
        return {
            'class_name': self.__class__.__name__,
            'tube_indices': self.tube_indices.tolist(),
            'cross_section_indices': self.cross_section_indices.tolist(),
            'pairings': self.pairings.tolist(),
        }

    def __str__(self):
        string = ""
        string += "Tube indices:\n"
        string += str(self.tube_indices) + "\n"
        string += "Cross-section indices:\n"
        string += str(self.cross_section_indices) + "\n"
        string += "Pairings:\n"
        string += str(self.pairings) + "\n"
        return string

class FixedTargetCrossSections(TargetCrossSections):
    
    def __init__(self, target_points, tube_indices, cross_section_indices, pairings):
        '''
        Args:
            target_points (torch.Tensor of shape (Ns, n_points_cs, 3)): tensor of the target points.
            Same as in the TargetCrossSections class.
        '''
        super().__init__(tube_indices, cross_section_indices, pairings)
        self.target_points = target_points
        
    def set_target_points(self, target_points):
        self.target_points = target_points
        
    def get_target_points_for_tube(self, tube_idx):
        """
        Get the target points for the tube with index tube_idx as a tensor of shape (n_target_cross_sections, N, 3), 
        where the first two values depend on tube_idx.
        """
        assert tube_idx in self.tube_indices, "The tube index must be in the tube_indices."
        # Find the positions of all the occurrences of tube_idx in the tube_indices
        return torch.stack([self.target_points[i] for i in self.target_cross_sections_id_for_tubes[tube_idx]], dim=0)
    
    def get_target_points(self):
        return self.target_points
    
    def to_dict(self):
        """Convert to JSON-serializable dictionary."""
        base_dict = super().to_dict()
        base_dict['target_points'] = self.target_points.tolist()
        return base_dict
    
    @classmethod
    def from_dict(cls, data):
        """Create instance from dictionary."""
        target_points = torch.tensor(data['target_points'])
        tube_indices = torch.tensor(data['tube_indices'])
        cross_section_indices = torch.tensor(data['cross_section_indices'])
        pairings = torch.tensor(data['pairings'])
        return cls(target_points, tube_indices, cross_section_indices, pairings)
    
class ConnectorTargetCrossSections(TargetCrossSections):
    
    def __init__(self, connectors, tubes, connector_to_tube_connectivity, pairings):
        '''
        Args:
            connectors (list of Connector objects): list of the connectors.
            tubes (list of CTube objects): list of the tubes.
            connector_to_tube_connectivity (list of list of pairs of int): List of lists of pairs of integers that specify the connectivity between connectors and tubes. The pair P located in [i][j] specifies the connector i is connected through arm j of the connector to the tube P[0] at the start (P[1] == 0) or end (P[1] == 1).
            Same as in the TargetCrossSections class.
        '''
        
        assert all([isinstance(connector, Connector) for connector in connectors]), "All connectors must be of type Connector."
        assert all([isinstance(tube, CTube) for tube in tubes]), "All tubes must be of type CTube."
        self.n_arm_per_connector = [connector.n_arms for connector in connectors]
        
        tube_indices = []
        cross_section_indices = []
        self.connector_to_tube_connectivity = connector_to_tube_connectivity
        # Each of the pairs added to the list will be [id_connector, id_arm]
        self.tube_to_arm = []
        for id_c, c_tmp in enumerate(self.connector_to_tube_connectivity):
            for id_a, a_tmp in enumerate(c_tmp):
                tube_indices.append(a_tmp[0])
                M_tmp = tubes[a_tmp[0]].M
                cross_section_indices.append(0 if a_tmp[1] == 0 else M_tmp - 1)
                self.tube_to_arm.append([id_c, id_a])

        self.connectors = connectors
        
        super().__init__(torch.tensor(tube_indices, dtype=torch.int32), torch.tensor(cross_section_indices, dtype=torch.int32), pairings)
        
    def get_target_points_for_tube(self, tube_idx):
        "Get the target points for the tube with index tube_idx."
        assert tube_idx in self.tube_indices, "The tube index must be in the tube_indices."
        local_indices = self.tube_idx_to_local_indices_index[tube_idx] # gives the indices in tube_to_arm for the tube_idx
        vertices = torch.stack([self.connectors[idx].get_arm_cross_section_vertices(arm_idx) for idx, arm_idx in [self.tube_to_arm[i] for i in local_indices]], dim=0) # shape (Ns, n_points_cs, 3)
        return vertices
    
    def get_target_points(self):
        return torch.cat([connector.get_arms_cross_section_vertices() for connector in self.connectors], dim=0)


def fix_end_cross_sections(tube, pairings=None):
    """
    Returns a FixedTargetCrossSections object for the first and last cross-sections of the tube.

    Parameters
    ----------
    tube : CTube
        The tube object for which to extract the end cross-sections.
    pairings : torch.Tensor, optional
        Tensor of allowed pairings between the indices of the target cross section and the candidate one in the tube.
        If None, uses get_pairings_exact(N) as default.

    Returns
    -------
    FixedTargetCrossSections
        Object containing the target points, tube indices, cross-section indices, and pairings for the ends.
    """
    ctube_vertices = tube.compute_vertices()
    M, N, _ = ctube_vertices.shape
    match_cross_section_indices = torch.tensor([0, M-1], dtype=torch.int64)
    match_tube_indices = torch.zeros(size=(2,), dtype=torch.int64)
    target_points = ctube_vertices[match_cross_section_indices]

    if pairings is None:
        pairings = get_pairings_exact(N)  # default
        pairings = pairings.repeat(2, 1, 1, 1)  # create two identical copies, one per cross-section
    elif pairings.dim() == 3:  # if a single copy is provided
        pairings = pairings.repeat(2, 1, 1, 1)  # create two identical copies, one per cross-section
    
    return FixedTargetCrossSections(target_points, match_tube_indices, match_cross_section_indices, pairings)

def generate_full_pairings_constant_cross_section_number(Ns, n_points_cs):
    '''
    Generate the pairings for the case when the number of points in the cross-section is constant; everyone is paired with everyone
    
    Args:
        Ns (int): number of cross sections.
        n_points_cs (int): number of points in the cross-section.
        
    Returns:
        pairings (torch.tensor of shape (Ns, n_pairings, n_points_cs, 2)): tensor of the allowed pairings between the indices of the target cross section [..., 0] and the candidate one in the tube [..., 1].
    '''
    n_pairings = n_points_cs
    pairings = torch.zeros((Ns, n_pairings, n_points_cs, 2), dtype=torch.int32)
    pairings[..., 0] = torch.arange(n_points_cs)
    for id_pairing in range(n_pairings):
        pairings[:, id_pairing, :, 1] = torch.roll(torch.arange(n_points_cs), shifts=id_pairing, dims=0)
    return pairings

def generate_full_pairings_with_flips_constant_cross_section_number(Ns, n_points_cs):
    '''
    Generate the pairings for the case when the number of points in the cross-section is constant; everyone is paired with everyone
    
    Args:
        Ns (int): number of cross sections.
        n_points_cs (int): number of points in the cross-section.
        
    Returns:
        pairings (torch.tensor of shape (Ns, 2*n_pairings, n_points_cs, 2)): tensor of the allowed pairings between the indices of the target cross section [..., 0] and the candidate one in the tube [..., 1].
    '''
    n_pairings = n_points_cs
    pairings = torch.zeros((Ns, 2*n_pairings, n_points_cs, 2), dtype=torch.int32)
    pairings[..., 0] = torch.arange(n_points_cs)
    for id_pairing in range(n_pairings):
        pairings[:, 2*id_pairing, :, 1] = torch.roll(torch.arange(n_points_cs), shifts=id_pairing, dims=0)
        pairings[:, 2*id_pairing+1, :, 1] = torch.roll(torch.flip(torch.arange(n_points_cs), dims=[0]), shifts=id_pairing, dims=0)
    return pairings

def generate_consistent_pairings_constant_cross_section_number(Ns, n_points_cs):
    '''
    Generate the pairings for the case when the number of points in the cross-section is constant; everyone is paired with its corresponding index
    
    Args:
        Ns (int): number of cross sections.
        n_points_cs (int): number of points in the cross-section.
        
    Returns:
        pairings (torch.tensor of shape (Ns, n_pairings, n_points_cs, 2)): tensor of the allowed pairings between the indices of the target cross section [..., 0] and the candidate one in the tube [..., 1].
    '''
    n_pairings = n_points_cs
    pairings = torch.zeros((Ns, n_pairings, n_points_cs, 2), dtype=torch.int32)
    pairings[..., 0] = torch.arange(n_points_cs)
    pairings[..., 1] = torch.arange(n_points_cs)
    return pairings
    