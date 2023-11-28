import pandas
from .residential_buildings import residential_buildings
from .churches import churches
from .barns import barns

def build_building_df(df):
    columns = ['menu', 'cat1', 'direction', 'row', 'col', 'is_diagonal', 'stories', 'weight', 'width', 'height','is_modular']
    building_dict = {col: [] for col in columns}

    for entry in df:
        for idx, el in enumerate(entry):
            building_dict[columns[idx]].append(el)

    building_df = pandas.DataFrame(building_dict)

    return building_df

def get_building_tiles(building_type):
    if building_type == 'churches':
        return build_building_df(churches)
    elif building_type == 'barns':
        return build_building_df(barns)
    else:
        return build_building_df(residential_buildings)

def get_building_cat2(building_type, row, col):
    if building_type == 'churches':
        return 'Church {}'.format(row * 3 + col + 1)
    elif building_type == 'barns':
        return 'Barn {}'.format(row * 3 + col + 1)
    else:
        return 'Building {}'.format(row * 4 + col + 1)
