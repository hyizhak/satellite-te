def _idx2coord(idx, x_sz, y_sz, offset):
    idx_ = idx - offset
    return (idx_ // y_sz, idx_ % y_sz)

def _coord2idx(coord, x_sz, y_sz, offset):
    return coord[0] * y_sz + coord[1] + offset

def _x_path(start_x, stop_x, y, x_sz, x_inc):
    path = []
    x = start_x
    while x != stop_x:
        next_x = (x + x_inc + x_sz) % x_sz
        path.append((
            (x, y), (next_x, y)
        ))
        x = next_x
    return path
        
def _y_path(start_y, stop_y, x, y_sz, y_inc):
    path = []
    y = start_y
    while y != stop_y:
        next_y = (y + y_inc + y_sz) % y_sz
        path.append((
            (x, y), (x, next_y)
        ))
        y = next_y
    return path

def _zigzag_path(src_coord, dst_coord, x_sz, y_sz, x_inc, y_inc):
    
    x = src_coord[0]
    y = src_coord[1]
    
    edges = []
       
    while x != dst_coord[0] and y != dst_coord[1]:
        next_x = (x + x_inc + x_sz) % x_sz
        next_y = (y + y_inc + y_sz) % y_sz
        edges.append((
            (x, y), (next_x, y)
        ))
        edges.append((
            (next_x, y), (next_x, next_y)
        ))
        x, y = next_x, next_y
        
    if x != dst_coord[0]:
        edges += _x_path(x, dst_coord[0], y, x_sz, x_inc)
        
    if y != dst_coord[1]:
        edges += _y_path(y, dst_coord[1], x, y_sz, y_inc)
    
    return edges

def k_shortest_path_loop_grid(src_idx, dst_idx, idx_offset, x_sz, y_sz, path_num):

    srcc = _idx2coord(src_idx, x_sz, y_sz, idx_offset)
    dstc = _idx2coord(dst_idx, x_sz, y_sz, idx_offset)
    
    x_inc = 1 if (dstc[0] + x_sz - srcc[0]) % x_sz <= (x_sz // 2) else -1
    y_inc = 1 if (dstc[1] + y_sz - srcc[1]) % y_sz <= (y_sz // 2) else -1
    
    paths = []
    paths.append(_zigzag_path(srcc, dstc, x_sz, y_sz, x_inc, y_inc))
    
    if srcc[0] != dstc[0] and srcc[1] != dstc[1]:
        
        x, y = srcc[0], srcc[1]
        y_limit = dstc[1]
        x_limit = (dstc[0] + x_sz - x_inc) % x_sz
        
        while x != x_limit and y != y_limit:
            x = (x + x_inc + x_sz) % x_sz
            y = (y + y_inc + y_sz) % y_sz
            
            if len(paths) >= path_num:
                break
            pathx = _x_path(srcc[0], x, srcc[1], x_sz, x_inc) + _zigzag_path((x, srcc[1]), dstc, x_sz, y_sz, x_inc, y_inc)
            paths.append(pathx)

            if len(paths) >= path_num:
                break
            pathy = _y_path(srcc[1], y, srcc[0], y_sz, y_inc) + _zigzag_path((srcc[0], y), dstc, x_sz, y_sz, x_inc, y_inc)
            paths.append(pathy)
            
        while len(paths) < path_num and x != x_limit:
            x = (x + x_inc + x_sz) % x_sz
            pathx = _x_path(srcc[0], x, srcc[1], x_sz, x_inc) + _zigzag_path((x, srcc[1]), dstc, x_sz, y_sz, x_inc, y_inc)
            paths.append(pathx)
                    
        while len(paths) < path_num and y != y_limit:
            y = (y + y_inc + y_sz) % y_sz
            pathy = _y_path(srcc[1], y, srcc[0], y_sz, y_inc) + _zigzag_path((srcc[0], y), dstc, x_sz, y_sz, x_inc, y_inc)
            paths.append(pathy)
        
    res = []
    for path in paths:
        node_list = [src_idx] + [_coord2idx(c2, x_sz, y_sz, idx_offset) for (c1, c2) in path]
        res.append(node_list)

    return res
    
    
def test():
    offset = 0
    x_sz = 10
    y_sz = 10

    def get_idx(c):
        return c[0] * y_sz + c[1] + offset
    
    c1 = (1, 4)
    c2 = (0, 0)

    paths = k_shortest_path_loop_grid(get_idx(c1), get_idx(c2), offset, x_sz, y_sz, 7)
    print(paths)
    
if __name__ == '__main__':
    test()
    
    