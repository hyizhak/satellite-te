from . import inter_shell as _inter_shell
from ..ism import InterShellMode as ISM
from copy import deepcopy
from ..user_node import generate_sat2user, generate_user2sat, generate_is_user


class StarlinkPathFormer(object):

    @staticmethod
    def create_metadata(sat_num, grdstation_num, G_intershell, ISL_intershell):
        meta = {
            "sat_num": sat_num,
            "grdstation_num": grdstation_num,
            "G_intershell": deepcopy(G_intershell),
            "ISL_intershell": deepcopy(ISL_intershell)
        }
        return meta
    
    def __init__(self, metadata, mode=ISM.GRD_STATION):
        self.metadata = metadata
        self.sat_num = metadata["sat_num"]
        self.grdstation_num = metadata["grdstation_num"]
        self.G_intershell = metadata["G_intershell"]
        self.ISL_intershell = metadata["ISL_intershell"]
        
        self.mode = mode
        
        # Generate utilities
        self.sat2user = generate_sat2user(self.sat_num, self.grdstation_num, mode)
        self.user2sat = generate_user2sat(self.sat_num, self.grdstation_num, mode)
        self.is_user = generate_is_user(self.sat_num, self.grdstation_num, mode)
        
    def _compute_path_grdstation(self, src, dst, path_num):
        sat_src = self.user2sat(src)
        sat_dst = self.user2sat(dst)
        
        sat_paths = _inter_shell.SPOnGrid(sat_src, sat_dst, self.G_intershell, None, ISM.GRD_STATION, path_num)
        
        return [[src] + path + [dst] for path in sat_paths]
        
    def compute_path(self, src, dst, path_num):
        if not self.is_user(src) or not self.is_user(dst):
            raise ValueError("Source or destination is not a user node")
        
        match self.mode:
            case ISM.GRD_STATION:
                return self._compute_path_grdstation(src, dst, path_num)
            case _:
                raise NotImplementedError()                
            