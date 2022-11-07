# Copyright (C) 2022  Nicolas Möser

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import pandas

road_tiles_dict = {
    (0, 0, 0): [('u', (2,3)), ('d', (2,3)), ('cost', 1.0)],
    (0, 0, 1): [('d', (1,4)), ('r', (1,4)), ('cost', 0.7)],
    (0, 0, 2): [('u', (2,4)), ('d', (2,3)), ('cost', 1.1)],
    (0, 1, 0): [('d', (2,3)), ('r', (2,3)), ('cost', 1.0)],
    (0, 1, 1): [('d', (2,4)), ('r', (1,4)), ('cost', 0.7)],
    (0, 1, 2): [('u', (2,3)), ('d', (2,3)), ('r', (2,3)), ('cost', 1.0)],
    (0, 2, 0): [('d', (1,4)), ('r', (1,4)), ('l', (1,4)), ('cost', 1.0)],
    (0, 2, 1): [('u', (2,3)), ('d', (2,3)), ('r', (1,4)), ('cost', 1.0)],
    (0, 2, 2): [('u', (2,3)), ('d', (2,3)), ('r', (2,3)), ('l', (2,3)), ('cost', 1.0)],
    (0, 3, 0): [('u', (1,3)), ('d', (2,3)), ('cost', 1.1)],
    (0, 3, 1): [('d', (1,3)), ('l', (1,4)), ('cost', 0.7)],
    (0, 3, 2): [('u', (2,3)), ('d', (2,3)), ('l', (1,4)), ('cost', 1.0)],
    (0, 0, 2, 1): [('u', (2,4)), ('d', (2,4)), ('cost', 1.2)],
    (0, 0, 2, 2): [('u', (2,4)), ('d', (1,3)), ('cost', 1.2)],
    (0, 3, 0, 1): [('u', (1,3)), ('d', (2,4)), ('cost', 1.2)],
    (0, 3, 0, 2): [('u', (1,3)), ('d', (1,3)), ('cost', 1.2)],

    (1, 0, 0): [('r', (2,3)), ('l', (2,3)), ('cost', 1.0)],
    (1, 0, 1): [('u', (1,4)), ('r', (1,4)), ('cost', 0.7)],
    (1, 0, 2): [('r', (2,3)), ('l', (2,4)), ('cost', 1.1)],
    (1, 1, 0): [('u', (2,3)), ('r', (2,3)), ('cost', 1.0)],
    (1, 1, 1): [('u', (1,4)), ('r', (2,4)), ('cost', 0.7)],
    (1, 1, 2): [('u', (2,3)), ('r', (2,3)), ('l', (2,3)), ('cost', 1.0)],
    (1, 2, 0): [('u', (1,4)), ('d', (1,4)), ('r', (1,4)), ('cost', 1.0)],
    (1, 2, 1): [('u', (1,4)), ('r', (2,3)), ('l', (2,4)), ('cost', 1.0)],
    # (1, 2, 2): [('u', (2,3)), ('d', (2,3)), ('r', (2,3)), ('l', (2,3)), ('cost', 1.0)],
    (1, 3, 0): [('r', (2,3)), ('l', (1,3)), ('cost', 1.1)],
    (1, 3, 1): [('d', (1,4)), ('r', (1,3)), ('cost', 0.7)],
    (1, 3, 2): [('u', (1,4)), ('r', (2,3)), ('l', (2,3)), ('cost', 1.0)],
    (1, 0, 2, 1): [('r', (1,3)), ('l', (2,4)), ('cost', 1.2)],
    (1, 0, 2, 2): [('r', (2,4)), ('l', (2,4)), ('cost', 1.2)],
    (1, 3, 0, 1): [('r', (1,3)), ('l', (1,3)), ('cost', 1.2)],
    (1, 3, 0, 2): [('r', (2,4)), ('l', (1,3)), ('cost', 1.2)],

    # (2, 0, 0): [('u', (2,3)), ('d', (2,3)), ('cost', 1.0)],
    (2, 0, 1): [('u', (1,4)), ('l', (1,4)), ('cost', 0.7)],
    (2, 0, 2): [('u', (2,3)), ('d', (1,3)), ('cost', 1.1)],
    (2, 1, 0): [('u', (2,3)), ('l', (2,3)), ('cost', 1.0)],
    (2, 1, 1): [('u', (1,3)), ('l', (1,4)), ('cost', 0.7)],
    (2, 1, 2): [('u', (2,3)), ('d', (2,3)), ('l', (2,3)), ('cost', 1.0)],
    (2, 2, 0): [('u', (1,4)), ('r', (1,4)), ('l', (1,4)), ('cost', 1.0)],
    (2, 2, 1): [('u', (2,3)), ('d', (2,3)), ('l', (1,4)), ('cost', 1.0)],
    # (2, 2, 2): [('u', (2,3)), ('d', (2,3)), ('r', (2,3)), ('l', (2,3)), ('cost', 1.0)],
    (2, 3, 0): [('u', (2,3)), ('d', (2,4)), ('cost', 1.1)],
    (2, 3, 1): [('u', (2,4)), ('r', (1,4)), ('cost', 0.7)],
    (2, 3, 2): [('u', (2,3)), ('d', (2,3)), ('r', (1,4)), ('cost', 1.0)],
    (2, 0, 2, 1): [('u', (1,3)), ('d', (1,3)), ('cost', 1.2)],
    (2, 0, 2, 2): [('u', (2,4)), ('d', (1,3)), ('cost', 1.2)],
    (2, 3, 0, 1): [('u', (1,3)), ('d', (2,4)), ('cost', 1.2)],
    (2, 3, 0, 2): [('u', (2,4)), ('d', (2,4)), ('cost', 1.2)],

    # (3, 0, 0): [('r', (2,3)), ('l', (2,3)), ('cost', 1.0)],
    (3, 0, 1): [('d', (1,4)), ('l', (1,4)), ('cost', 0.7)],
    (3, 0, 2): [('r', (1,3)), ('l', (2,3)), ('cost', 1.1)],
    (3, 1, 0): [('d', (2,3)), ('l', (2,3)), ('cost', 1.0)],
    (3, 1, 1): [('d', (1,4)), ('l', (1,3)), ('cost', 0.7)],
    (3, 1, 2): [('d', (2,3)), ('r', (2,3)), ('l', (2,3)), ('cost', 1.0)],
    (3, 2, 0): [('u', (1,4)), ('d', (1,4)), ('l', (1,4)), ('cost', 1.0)],
    (3, 2, 1): [('d', (1,4)), ('r', (2,3)), ('l', (2,3)), ('cost', 1.0)],
    # (3, 2, 2): [('u', (2,3)), ('d', (2,3)), ('r', (2,3)), ('l', (2,3)), ('cost', 1.0)],
    (3, 3, 0): [('r', (2,4)), ('l', (2,3)), ('cost', 1.1)],
    (3, 3, 1): [('u', (1,4)), ('l', (2,4)), ('cost', 0.7)],
    (3, 3, 2): [('u', (1,4)), ('r', (2,3)), ('l', (2,3)), ('cost', 1.0)],
    (3, 0, 2, 1): [('r', (1,3)), ('l', (1,3)), ('cost', 1.2)],
    (3, 0, 2, 2): [('r', (1,3)), ('l', (2,4)), ('cost', 1.2)],
    (3, 3, 0, 1): [('r', (2,4)), ('l', (1,3)), ('cost', 1.2)],
    (3, 3, 0, 2): [('r', (2,4)), ('l', (2,4)), ('cost', 1.2)],
}

def build_road_tile_df():
    road_tile_df_dict = {
        'u': [],
        'd': [],
        'r': [],
        'l': [],
        'direction': [],
        'row': [],
        'col': [],
        'n_connections': [],
        'cost': [],
        'variant': []
    }
    for key, value in road_tiles_dict.items():
        road_tile_df_dict['direction'].append(key[0])
        road_tile_df_dict['row'].append(key[1])
        road_tile_df_dict['col'].append(key[2])
        if len(key) == 4:
            road_tile_df_dict['variant'].append(key[3])
        else:
            road_tile_df_dict['variant'].append(0)
        
        u_val = None
        d_val = None
        r_val = None
        l_val = None
        cost = 1.0

        n_connections = 0
        for entry in value:
            if entry[0] == 'u':
                u_val = entry[1]
                n_connections += 1
            if entry[0] == 'd':
                d_val = entry[1]
                n_connections += 1
            if entry[0] == 'r':
                r_val = entry[1]
                n_connections += 1
            if entry[0] == 'l':
                l_val = entry[1]
                n_connections += 1
            if entry[0] == 'cost':
                cost = entry[1]

        road_tile_df_dict['u'].append(u_val)
        road_tile_df_dict['d'].append(d_val)
        road_tile_df_dict['r'].append(r_val)
        road_tile_df_dict['l'].append(l_val)
        road_tile_df_dict['n_connections'].append(n_connections)
        road_tile_df_dict['cost'].append(cost)

    return pandas.DataFrame(road_tile_df_dict)
