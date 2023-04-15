from collections import OrderedDict
from . import fortress_italy
from . import cold_war

available_profiles = OrderedDict([
    ('Cold War', 'cold_war'),
    ('Fortress Italy', 'fortress_italy')
])

def get_building_tiles(building_type, profile='cold_war'):
    if profile == 'cold_war':
        return cold_war.get_building_tiles(building_type)
    elif profile == 'fortress_italy':
        return fortress_italy.get_building_tiles(building_type)

def get_building_cat2(building_type, row, col, profile='cold_war'):
    if profile == 'cold_war':
        return cold_war.get_building_cat2(building_type, row, col)
    elif profile == 'fortress_italy':
        return fortress_italy.get_building_cat2(building_type, row, col)
