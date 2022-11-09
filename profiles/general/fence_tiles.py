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

fence_tiles_dict = {
    (0, 0, 0): [('r', (2,3)), ('l', (2,3)), ('cost', 1.0)],
    (0, 0, 1): [('ur', (2,3)), ('dl', (2,3)), ('cost', 1.0)],
    (0, 0, 2): [('u', (2,3)), ('l', (2,3)), ('cost', 1.0)],
    (0, 1, 0): [('ul', (2,3)), ('dl', (2,3)), ('cost', 1.0)],
    (0, 1, 1): [('u', (2,3)), ('dl', (2,3)), ('cost', 1.0)],
    (0, 1, 2): [('u', (2,3)), ('dr', (2,3)), ('cost', 1.0)],
    (0, 2, 0): [('d', (2,3)), ('r', (2,3)), ('l', (2,3)), ('cost', 1.0)],
    (0, 2, 1): [('ur', (2,3)), ('dr', (2,3)), ('dl', (2,3)), ('cost', 1.0)],
    (0, 2, 2): [('u', (2,3)), ('d', (2,3)), ('r', (2,3)), ('r', (2,3)), ('cost', 1.0)],
    (0, 3, 0): [('ur', (2,3)), ('ul', (2,3)), ('dr', (2,3)), ('dl', (2,3)), ('cost', 1.0)],

    (1, 0, 0): [('u', (2,3)), ('d', (2,3)), ('cost', 1.0)],
    (1, 0, 1): [('ul', (2,3)), ('dr', (2,3)), ('cost', 1.0)],
    (1, 0, 2): [('d', (2,3)), ('l', (2,3)), ('cost', 1.0)],
    (1, 1, 0): [('dr', (2,3)), ('dl', (2,3)), ('cost', 1.0)],
    (1, 1, 1): [('dr', (2,3)), ('l', (2,3)), ('cost', 1.0)],
    (1, 1, 2): [('ur', (2,3)), ('l', (2,3)), ('cost', 1.0)],
    (1, 2, 0): [('u', (2,3)), ('d', (2,3)), ('r', (2,3)), ('cost', 1.0)],
    (1, 2, 1): [('ur', (2,3)), ('ul', (2,3)), ('dr', (2,3)), ('cost', 1.0)],

    (2, 0, 2): [('d', (2,3)), ('r', (2,3)), ('cost', 1.0)],
    (2, 1, 0): [('ur', (2,3)), ('dr', (2,3)), ('cost', 1.0)],
    (2, 1, 1): [('ur', (2,3)), ('d', (2,3)), ('cost', 1.0)],
    (2, 1, 2): [('ul', (2,3)), ('d', (2,3)), ('cost', 1.0)],
    (2, 2, 0): [('u', (2,3)), ('r', (2,3)), ('l', (2,3)), ('cost', 1.0)],
    (2, 2, 1): [('ur', (2,3)), ('ul', (2,3)), ('dl', (2,3)), ('cost', 1.0)],

    (3, 0, 2): [('u', (2,3)), ('r', (2,3)), ('cost', 1.0)],
    (3, 1, 0): [('ur', (2,3)), ('ul', (2,3)), ('cost', 1.0)],
    (3, 1, 1): [('ul', (2,3)), ('r', (2,3)), ('cost', 1.0)],
    (3, 1, 2): [('dl', (2,3)), ('r', (2,3)), ('cost', 1.0)],
    (3, 2, 0): [('u', (2,3)), ('d', (2,3)), ('l', (2,3)), ('cost', 1.0)],
    (3, 2, 1): [('ul', (2,3)), ('dr', (2,3)), ('dl', (2,3)), ('cost', 1.0)],

}

def build_fence_tile_df():
    fence_tile_df_dict = {
        'u': [],
        'd': [],
        'r': [],
        'l': [],
        'ur': [],
        'ul': [],
        'dr': [],
        'dl': [],
        'direction': [],
        'row': [],
        'col': [],
        'n_connections': [],
        'cost': [],
        'variant': []
    }
    for key, value in fence_tiles_dict.items():
        fence_tile_df_dict['direction'].append(key[0])
        fence_tile_df_dict['row'].append(key[1])
        fence_tile_df_dict['col'].append(key[2])
        if len(key) == 4:
            fence_tile_df_dict['variant'].append(key[3])
        else:
            fence_tile_df_dict['variant'].append(0)

        u_val = None
        d_val = None
        r_val = None
        l_val = None
        ur_val = None
        ul_val = None
        dr_val = None
        dl_val = None
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
            if entry[0] == 'ur':
                ur_val = entry[1]
                n_connections += 1
            if entry[0] == 'ul':
                ul_val = entry[1]
                n_connections += 1
            if entry[0] == 'dr':
                dr_val = entry[1]
                n_connections += 1
            if entry[0] == 'dl':
                dl_val = entry[1]
                n_connections += 1
            if entry[0] == 'cost':
                cost = entry[1]

        fence_tile_df_dict['u'].append(u_val)
        fence_tile_df_dict['d'].append(d_val)
        fence_tile_df_dict['r'].append(r_val)
        fence_tile_df_dict['l'].append(l_val)
        fence_tile_df_dict['ur'].append(ur_val)
        fence_tile_df_dict['ul'].append(ul_val)
        fence_tile_df_dict['dr'].append(dr_val)
        fence_tile_df_dict['dl'].append(dl_val)
        fence_tile_df_dict['n_connections'].append(n_connections)
        fence_tile_df_dict['cost'].append(cost)

    return pandas.DataFrame(fence_tile_df_dict)
