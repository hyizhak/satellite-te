from dataclasses import dataclass

@dataclass
class OrbitParams:
    GrdStationNum: int
    Offset5: int
    graph_node_num: int
    isl_cap: int
    uplink_cap: int
    downlink_cap: int
    ism: str
