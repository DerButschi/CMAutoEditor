from collections import OrderedDict
from . import fortress_italy
from . import cold_war
from . import shock_force_2
from . import black_sea
import numpy as np
from skimage.draw import polygon2mask

available_profiles = OrderedDict([
    ('Black Sea', 'black_sea'),
    ('Cold War', 'cold_war'),
    ('Fortress Italy', 'fortress_italy'),
    ('Shock Force 2', 'shock_force_2')
])

def get_building_tiles(building_type, profile='cold_war'):
    if profile == 'cold_war':
        return cold_war.get_building_tiles(building_type)
    elif profile == 'fortress_italy':
        return fortress_italy.get_building_tiles(building_type)
    elif profile == 'shock_force_2':
        return shock_force_2.get_building_tiles(building_type)
    elif profile == 'black_sea':
        return black_sea.get_building_tiles(building_type)

def get_building_cat2(building_type, row, col, profile='cold_war'):
    if profile == 'cold_war':
        return cold_war.get_building_cat2(building_type, row, col)
    elif profile == 'fortress_italy':
        return fortress_italy.get_building_cat2(building_type, row, col)
    elif profile == 'shock_force_2':
        return shock_force_2.get_building_cat2(building_type, row, col)
    elif profile == 'black_sea':
        return black_sea.get_building_cat2(building_type, row, col)

def get_building_tokens(building_type, profile='cold_war'):
    building_tiles = get_building_tiles(building_type, profile)

    building_tokens = OrderedDict()

    for name, group in building_tiles.groupby(by=['is_diagonal', 'width', 'height']):
        width = name[1]
        height = name[2]
        is_diagonal = name[0]
        building_idx = group.index[0]

        token_dict = {}


        if not is_diagonal:
            pattern = np.ones((height, width), dtype=int)
        else:
            polygon = [
                (height - 0.5, -0.5),
                (height + width - 0.5, width - 0.5),
                (width - 0.5, height + width - 0.5),
                (-0.5, height - 0.5)
            ]
            pattern = np.zeros((height + width, height + width), dtype=float)
            mask = polygon2mask((pattern.shape), polygon)
            pattern[mask] = 1
            for row, col in np.argwhere(pattern == 1):
                if row > 0 and pattern[row - 1, col] == 0:
                    pattern[row, col] = 0.5
                elif row < pattern.shape[0] - 1 and pattern[row + 1, col] == 0:
                    pattern[row, col] = 0.5
                elif col > 0 and pattern[row, col - 1] == 0:
                    pattern[row, col] = 0.5
                elif col < pattern.shape[1] - 1 and pattern[row, col + 1] == 0:
                    pattern[row, col] = 0.5


                
                                    

            a = 1

        token_dict['pattern'] = pattern
        token_dict['has_modular'] = group['is_modular'].any()
        token_dict['is_diagonal'] = is_diagonal

        building_tokens[building_idx] = token_dict

    return building_tokens
