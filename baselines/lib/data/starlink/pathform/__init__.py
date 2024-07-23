from . import inter_shell as _inter_shell
from ..ism import InterShellMode as ISM
from copy import deepcopy
from ..user_node import generate_sat2user, generate_user2sat, generate_is_user


class StarlinkPathFormer(object):

    @staticmethod
    def create_metadata(sat_num, grdstation_num, G_intershell, ISL_intershell, reduceF):
        meta = {
            "sat_num": sat_num,
            "grdstation_num": grdstation_num,
            "G_intershell": deepcopy(G_intershell),
            "ISL_intershell": deepcopy(ISL_intershell),
            "reduceF": reduceF
        }
        return meta
    
    @staticmethod
    def get_pathdict_id(path_num, mode:ISM):
        return f"num-{path_num}_ism-{mode.value}"
    
    def __init__(self, metadata, mode=ISM.GRD_STATION):
        self.metadata = metadata
        self.sat_num = metadata["sat_num"]
        self.grdstation_num = metadata["grdstation_num"]
        self.G_intershell = metadata["G_intershell"]
        self.ISL_intershell = metadata["ISL_intershell"]
        self.reduceF = metadata["reduceF"]
        
        self.mode = mode
        
        # Generate utilities
        self.sat2user = generate_sat2user(self.sat_num, self.grdstation_num, mode)
        self.user2sat = generate_user2sat(self.sat_num, self.grdstation_num, mode)
        self.is_user = generate_is_user(self.sat_num, self.grdstation_num, mode)
        self.is_sat = lambda x: x < self.sat_num
        
        if self.mode == ISM.GRD_STATION:
            self.is_grdstation = lambda x: x >= self.sat_num and x < self.sat_num + self.grdstation_num
            self.grdstation2node = lambda x: x + self.sat_num
            self.linked_grd = None
            for i in range(0, self.grdstation_num):
                grdstation_sat = _inter_shell.SatOverGrdStation(i, self.G_intershell)
                if grdstation_sat != None:
                    self.linked_grd = self.grdstation2node(i)
                    self.linked_sat = grdstation_sat
            assert self.linked_grd != None, "No ground station linked to a satellite"
        
    def _compute_path_grdstation_none_user(self, src, dst, path_num):
        assert src < self.sat_num + self.grdstation_num and dst < self.sat_num + self.grdstation_num, "Source or destination is not a user node"
        if self.is_sat(src) and self.is_sat(dst):
            return _inter_shell.SPOnGrid(src, dst, self.G_intershell, None, ISM.GRD_STATION, path_num, self.reduceF)
        elif self.is_sat(src):
            sat_paths = _inter_shell.SPOnGrid(src, self.linked_sat, self.G_intershell, None, ISM.GRD_STATION, path_num, self.reduceF)
            tail = [dst] if dst == self.linked_grd else [self.linked_grd, dst]
            paths = [path + tail for path in sat_paths]
            return paths
        elif self.is_sat(dst):
            sat_paths = _inter_shell.SPOnGrid(self.linked_sat, dst, self.G_intershell, None, ISM.GRD_STATION, path_num, self.reduceF)
            head = [src] if src == self.linked_grd else [src, self.linked_grd]
            paths = [head + path for path in sat_paths]
            return paths
        else:
            return [[src, dst]]
               
        
    def _compute_path_grdstation_user(self, src, dst, path_num):
        sat_src = self.user2sat(src)
        sat_dst = self.user2sat(dst)
        
        sat_paths = _inter_shell.SPOnGrid(sat_src, sat_dst, self.G_intershell, None, ISM.GRD_STATION, path_num, self.reduceF)
        
        return [[src] + path + [dst] for path in sat_paths]
    
    def _compute_pathdict_grdstation(self, path_num):
        path_dict = {}
        for src in range(self.sat_num + self.grdstation_num):
            for dst in range(self.sat_num + self.grdstation_num):
                if src == dst:
                    continue
                # print(f"Computing path from {src} to {dst}")
                path_dict[(src, dst)] = self._compute_path_grdstation_none_user(src, dst, path_num)
        
        for none_user in range(self.sat_num + self.grdstation_num):
            for sat in range(self.sat_num):
                user = self.sat2user(sat)
                # print(f"Computing path from no_user {no_user} <-> user {user}")
                if sat == none_user:
                    path_dict[(none_user, user)] = [[none_user, user]]
                    path_dict[(user, none_user)] = [[user, none_user]]
                else:
                    path_dict[(none_user, user)] = [
                        path + [user] for path in path_dict[(none_user, sat)]
                    ]
                    path_dict[(user, none_user)] = [
                        [user] + path for path in path_dict[(sat, none_user)]
                    ]
        for sat1 in range(self.sat_num):
            for sat2 in range(self.sat_num):
                if sat1 == sat2:
                    continue
                user1 = self.sat2user(sat1)
                user2 = self.sat2user(sat2)
                # print(f"Computing path from user {user1} <-> user {user2}")
                path_dict[(user1, user2)] = [
                    [user1] + path + [user2] for path in path_dict[(sat1, sat2)]
                ]
                path_dict[(user2, user1)] = [
                    [user2] + path + [user1] for path in path_dict[(sat2, sat1)]
                ]
        
        node_num = self.sat_num * 2 + self.grdstation_num
        assert len(path_dict) == node_num * (node_num - 1)
        
        no_cycle_path_dcit = {}
        for pair, paths in path_dict.items():
            no_cycle_path_dcit[pair] = [self._remove_cycles(path) for path in paths]
        
        return no_cycle_path_dcit
        
    def compute_path(self, src, dst, path_num):
        if not self.is_user(src) or not self.is_user(dst):
            raise ValueError("Source or destination is not a user node")
        
        match self.mode:
            case ISM.GRD_STATION:
                return self._compute_path_grdstation_user(src, dst, path_num)
            case _:
                raise NotImplementedError()
            
    def compute_pathdict(self, path_num):
        
        match self.mode:
            case ISM.GRD_STATION:
                return self._compute_pathdict_grdstation(path_num)
            case _:
                raise NotImplementedError()
            
    def _remove_cycles(self, path):
        stack = []
        visited = set()
        for node in path:
            if node in visited:
                # remove elements from this cycle
                while stack[-1] != node:
                    visited.remove(stack[-1])
                    stack = stack[:-1]
            else:
                stack.append(node)
                visited.add(node)
        return stack
        