from .ism import InterShellMode as ISM

def generate_sat2user(satellite_num, grdstation_num, mode:ISM):
    match mode:
        case ISM.GRD_STATION:
            return lambda sat: sat + satellite_num + grdstation_num
        case _:
            raise NotImplementedError()

def generate_user2sat(satellite_num, grdstation_num, mode:ISM):
    match mode:
        case ISM.GRD_STATION:
            return lambda user: user - satellite_num - grdstation_num

def generate_is_user(satellite_num, grdstation_num, mode:ISM):
    match mode:
        case ISM.GRD_STATION:
            return lambda x: x >= satellite_num + grdstation_num and x < satellite_num * 2 + grdstation_num
        case _:
            raise NotImplementedError()
