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

from time import sleep
import pyautogui
import numpy as np
import pandas
import argparse
import keyboard
import os
import PySimpleGUI as sg
import sys

# constants:
UPPER_LEFT_SQUARE = pyautogui.Point(233,52)
# UPPER_RIGHT_SQUARE = pyautogui.Point(1882,52)
# LOWER_LEFT_SQUARE = pyautogui.Point(234,996)
LOWER_RIGHT_SQUARE = pyautogui.Point(633,452)

SQUARE_SIZE_X = 16
SQUARE_SIZE_Y = 16

START_N_SQUARES_X = 40
START_N_SQUARES_Y = 40

PAGE_N_SQUARES_X = 26
PAGE_N_SQUARES_Y = 26

START_HEIGHT = 20

POS_HORIZONTAL_PLUS = pyautogui.Point(764, 10)
POS_HORIZONTAL_MINUS = pyautogui.Point(764, 26)
POS_HORIZONTAL_PLUS2 = pyautogui.Point(874, 10)
POS_HORIZONTAL_MINUS2 = pyautogui.Point(874, 27)

POS_VERTICAL_PLUS = pyautogui.Point(1014, 10)
POS_VERTICAL_MINUS = pyautogui.Point(903, 10)

MENU_DICT = {
    'Ground 1': pyautogui.Point(113, 107),
    'Ground 2': pyautogui.Point(105, 123),
    'Ground 3': pyautogui.Point(105, 144),
    'Brush': pyautogui.Point(107, 165),
    'Foliage': pyautogui.Point(110, 185),
    'Roads': pyautogui.Point(107, 203),
    "Walls/Fences": pyautogui.Point(105, 224),
    "Independent Buildings": pyautogui.Point(105, 264),
    "Modular Buildings": pyautogui.Point(105, 245),
    "Flavor Objects 1": pyautogui.Point(105, 285),
    "Flavor Objects 2": pyautogui.Point(105, 306),
    'Water': pyautogui.Point(81, 438),
    'Plow NS': pyautogui.Point(135, 552),
    'Plow EW': pyautogui.Point(188, 552),
    'Crop 1': pyautogui.Point(26, 383),
    'Crop 2': pyautogui.Point(87, 383),
    'Crop 3': pyautogui.Point(135, 383),
    'Crop 4': pyautogui.Point(188, 383),
    'Crop 5': pyautogui.Point(26, 440),
    'Crop 6': pyautogui.Point(87, 440),
    'Tree A': pyautogui.Point(110, 381),
    'Tree B': pyautogui.Point(180, 380),
    'Tree C': pyautogui.Point(39, 438),
    'Tree D': pyautogui.Point(114, 438),
    'Tree E': pyautogui.Point(183, 438),
    'Tree F': pyautogui.Point(39, 498),
    'Tree G': pyautogui.Point(114, 498),
    'Tree H': pyautogui.Point(183, 498),
    'Bush A': pyautogui.Point(39, 555),
    'Bush B': pyautogui.Point(114, 555),
    'Bush C': pyautogui.Point(183, 555),
    'density 1': pyautogui.Point(38, 617),
    'density 2': pyautogui.Point(110, 617),
    'density 3': pyautogui.Point(180, 617),
    'density 4': pyautogui.Point(38, 657),
    'Grass': pyautogui.Point(189, 438),
    'Flowers': pyautogui.Point(137, 498),
    'Grass T': pyautogui.Point(191, 498),
    'Grass TY': pyautogui.Point(27, 554),
    'Weeds': pyautogui.Point(80, 554),
    'Grass XT': pyautogui.Point(135, 554),
    'Grass XTY': pyautogui.Point(191, 554),
    'Dirt': pyautogui.Point(27, 383),
    'Pavement 1': pyautogui.Point(81, 495),
    'Direction 1': pyautogui.Point(248,17),
    'Direction 2': pyautogui.Point(278,17),
    'Direction 3': pyautogui.Point(308,17),
    'Direction 4': pyautogui.Point(338,17),
    'Road Tile 1': pyautogui.Point(36,615),
    'Road Tile 2': pyautogui.Point(108,615),
    'Road Tile 3': pyautogui.Point(182,615),
    'Road Tile 4': pyautogui.Point(36,659),
    'Road Tile 5': pyautogui.Point(108,659),
    'Road Tile 6': pyautogui.Point(182,659),
    'Road Tile 7': pyautogui.Point(36,696),
    'Road Tile 8': pyautogui.Point(108,696),
    'Road Tile 9': pyautogui.Point(182,696),
    'Road Tile 10': pyautogui.Point(36,740),
    'Road Tile 11': pyautogui.Point(108,740),
    'Road Tile 12': pyautogui.Point(182,740),
    'Paved 1': pyautogui.Point(36,440),
    'Paved 2': pyautogui.Point(108,440),
    'Foot Path': pyautogui.Point(183,440),
    'Gravel Road': pyautogui.Point(183,383),
    'Marsh': pyautogui.Point(189, 381),
    'Deep Ford': pyautogui.Point(27, 495),
    'Dirt Red': pyautogui.Point(83, 383),
    'Gravel': pyautogui.Point(27, 554),
    'Sand': pyautogui.Point(134, 440),
    'Railroad': pyautogui.Point(36, 497),
    'Stream': pyautogui.Point(108, 497),
    'Stone': pyautogui.Point(81, 381),
    'Tall Stone': pyautogui.Point(135, 381),
    'Brick': pyautogui.Point(189, 381),
    'Tall Brick': pyautogui.Point(27, 441),
    'Rural Stone': pyautogui.Point(81, 441),
    'Hedge': pyautogui.Point(135, 441),
    'Low Bocage': pyautogui.Point(189, 441),
    'Wood Fence': pyautogui.Point(27, 498),
    'Wire Fence': pyautogui.Point(81, 498),
    'Picket': pyautogui.Point(135, 498),
    'Sticks': pyautogui.Point(189, 498),
    'Lt Forest': pyautogui.Point(27, 383),
    'Deep Marsh': pyautogui.Point(27, 440),
    'Brush_brush': pyautogui.Point(110, 383),
    'Ground 2 Paved 2': pyautogui.Point(135, 497),
    'Cobblestone': pyautogui.Point(190, 497),
    'House': pyautogui.Point(110, 383),
    'Barn': pyautogui.Point(36, 440),
    '1 Story': pyautogui.Point(81, 383),
    '2 Story': pyautogui.Point(134, 383),
    '3 Story': pyautogui.Point(189, 383),
    'Building 1': pyautogui.Point(27, 617),
    'Building 2': pyautogui.Point(80, 617),
    'Building 3': pyautogui.Point(135, 617),
    'Building 4': pyautogui.Point(188, 617),
    'Building 5': pyautogui.Point(27, 657),
    'Building 6': pyautogui.Point(80, 657),
    'Building 7': pyautogui.Point(135, 657),
    'Building 8': pyautogui.Point(188, 657),
    'Building 9': pyautogui.Point(27, 699),
    'Building 10': pyautogui.Point(80, 699),
    'Building 11': pyautogui.Point(135, 699),
    'Building 12': pyautogui.Point(188, 699),
    'Building 13': pyautogui.Point(27, 740),
    'Building 14': pyautogui.Point(80, 740),
    'Building 15': pyautogui.Point(135, 740),
    'Building 16': pyautogui.Point(188, 740),
    'Gravestone': pyautogui.Point(188, 441),
    'Junk': pyautogui.Point(27, 498),
    'Pallet': pyautogui.Point(188, 498),
    'Bin': pyautogui.Point(188, 384),
    'Shed': pyautogui.Point(188, 384),
    'Shelter': pyautogui.Point(27, 438),
    'Fountain': pyautogui.Point(135, 440),
    'Pond': pyautogui.Point(81, 554),
    'Bench': pyautogui.Point(135, 383),
    'Roadside': pyautogui.Point(191, 554),
    'Object 1': pyautogui.Point(36, 614),
    'Object 2': pyautogui.Point(110, 614),
    'Object 3': pyautogui.Point(185, 614),
    'Object 4': pyautogui.Point(36, 657),
    'Object 5': pyautogui.Point(110, 657),
    'Object 6': pyautogui.Point(185, 657),
    'Object 7': pyautogui.Point(36, 701),
    'Object 8': pyautogui.Point(110, 701),
    'Object 9': pyautogui.Point(185, 701),
    'Dirt Road': pyautogui.Point(110, 382),

}

GROUND_2_DICT = {
    'Water': pyautogui.Point(81, 438),
    'Plow NS': pyautogui.Point(135, 552),
    'Plow EW': pyautogui.Point(188, 552),
}

GROUND_3_DICT = {
    'Crop 1': pyautogui.Point(26, 383),
    'Crop 2': pyautogui.Point(87, 383),
    'Crop 3': pyautogui.Point(135, 383),
    'Crop 4': pyautogui.Point(188, 383),
    'Crop 5': pyautogui.Point(26, 440),
    'Crop 6': pyautogui.Point(87, 440),
}

FOLIAGE_DICT = {
    'Tree A': pyautogui.Point(110, 381),
    'Tree B': pyautogui.Point(180, 380),
    'Tree C': pyautogui.Point(39, 438),
    'Tree D': pyautogui.Point(114, 438),
    'Tree E': pyautogui.Point(183, 438),
    'Tree F': pyautogui.Point(39, 498),
    'Tree G': pyautogui.Point(114, 498),
    'Tree H': pyautogui.Point(183, 498),
    'Bush A': pyautogui.Point(39, 555),
    'Bush B': pyautogui.Point(114, 555),
    'Bush C': pyautogui.Point(183, 555),
}

FOLIAGE_DENSITY_DICT = {
    0: pyautogui.Point(38, 617),
    1: pyautogui.Point(110, 617),
    2: pyautogui.Point(180, 617),
    3: pyautogui.Point(38, 657),
}

TILE_PAGE_TO_ARROW_DICT = {
    0: pyautogui.Point(248,17),
    1: pyautogui.Point(278,17),
    2: pyautogui.Point(308,17),
    3: pyautogui.Point(338,17),
}

ROAD_TYPE_DICT = {
    'Paved 1': pyautogui.Point(36,440),
    'Paved 2': pyautogui.Point(108,440),
    'Foot Path': pyautogui.Point(183,440),
}

TILE_ROW_DICT = {
    0: 615,
    1: 659,
    2: 696,
    3: 740
}

TILE_COL_DICT = {
    0: 36,
    1: 108,
    2: 182,
}
POS_VERTICAL_PLUS2 = pyautogui.Point(1014, 27)
POS_VERTICAL_MINUS2 = pyautogui.Point(903, 27)

pyautogui.PAUSE = 0.05  # 0.12 almost!! works

def set_height(current_height, target_height):
    if current_height == target_height:
        return
    elif current_height < target_height:
        n_diff = target_height - current_height
        # pyautogui.press('+', presses=n_diff, interval=0.5)
        for i in range(n_diff):
            keyboard.send('+')
            sleep(0.1)
    else:
        n_diff = current_height - target_height
        print(n_diff)
        # pyautogui.press('-', presses=n_diff, interval=0.5)
        for i in range(n_diff):
            keyboard.send('-')
            sleep(0.1)


    # sleep(1)

def process_segment(grid, start_height):
    values = grid.z.sort_values().unique()
    min_height = grid[grid.z >= 0].z.min()
    
    if np.isnan(min_height):
        return start_height
    
    set_height(start_height, min_height)
    height = min_height

    for val in values:
        grid_extract = grid[grid.z == val]
        # indices0, indices1 = np.where(grid == val)

        set_height(height, val)

        for ridx, row in grid_extract.iterrows():
            # idx0 = indices0[i]
            # idx1 = indices1[i]
            idx0 = row.x
            idx1 = row.y
            x_pos = int(idx0 * SQUARE_SIZE_X + UPPER_LEFT_SQUARE.x)
            y_pos = int(LOWER_RIGHT_SQUARE.y - idx1 * SQUARE_SIZE_Y)
            height = val

            pyautogui.click(x=x_pos, y=y_pos)

    return height

def set_n_squares(start_n_x, start_n_y, n_x, n_y, init):
    n_clicks_x = abs(int((start_n_x - n_x) / 2))
    n_clicks_y = abs(int((start_n_y - n_y) / 2))

    for i in range(n_clicks_x):
        if n_x <= start_n_x:
            if not init:
                pyautogui.click(POS_HORIZONTAL_PLUS2, interval=0.05)    
            pyautogui.click(POS_HORIZONTAL_MINUS, interval=0.05)
        else:
            pyautogui.click(POS_HORIZONTAL_PLUS, interval=0.05)
            if not init:
                pyautogui.click(POS_HORIZONTAL_MINUS2, interval=0.05)    
    
    for i in range(n_clicks_y):
        if n_y <= start_n_y:
            if not init:
                pyautogui.click(POS_VERTICAL_PLUS2, interval=0.05)    
            pyautogui.click(POS_VERTICAL_MINUS, interval=0.05)
        else:
            pyautogui.click(POS_VERTICAL_PLUS, interval=0.05)
            if not init:
                pyautogui.click(POS_VERTICAL_MINUS2, interval=0.05)    

def display_gui():
    # Construct window layout
    layout = [
        [sg.Titlebar('CMAutoEditor')],
        [sg.Text('You are about to start CMAutoEditor.')],
        [sg.Text('If you haven\'t done so yet, open up the CM Scenario Editor, go to map->Elevation and click \'Direct\'.')],
        [sg.Text('Make sure the map size is 320m x 320m.')],
        [sg.Text('Once you are ready to start click \'Start CMAutoEditor\'.')], 
        [sg.Text('During the countdown switch back to the CM Scenario Editor.')],
        [sg.Text('In case something goes wrong, move the mouse cursor to one of the screen corners.')],
        [sg.Text('')],
        [sg.Text('Select file: ')], 
        [sg.Input(), sg.FileBrowse(key='filepath', file_types=(('CSV files', '*.csv'),))],
        [sg.Text('Countdown: '), sg.InputCombo(key='countdown',values=[5, 10, 15, 20, 25, 30], default_value=10)],
        [sg.Text(text='', key='error_text')],
        [sg.Push(), sg.Button('Start CMAutoEditor', key='start'), sg.Exit(), sg.Push()]]

    # Create window with layout
    window = sg.Window('CMAutoEditor', layout)
    
    # Loop until window needs closing
    start = False
    while True:
        # Read UI inputs
        event, values = window.read()
        
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        
        if event == 'start':
            if values['filepath'] == '' or values['filepath'] is None:
                window['error_text'].update('Select a file before starting')
            else:
                start = True
                break

    window.close()
    # Start editor with UI inputs
    if start and values['filepath'] != '' and values['filepath'] != None:
        start_editor(values['filepath'], values['countdown'])

            
def set_roads(df):
    # for name, group in df.groupby(by='id'):
    #     prev_x = None
    #     prev_y = None
    #     for ridx, row in group.iterrows():
    #         x_pos = int(row.x * SQUARE_SIZE_X + UPPER_LEFT_SQUARE.x)
    #         y_pos = int(LOWER_RIGHT_SQUARE.y - row.y * SQUARE_SIZE_Y)
    #         if prev_x is not None and prev_y is not None:
    #             pyautogui.click(x=prev_x, y=prev_y, interval=0.5)
    #             pyautogui.click(x=x_pos, y=y_pos, interval=0.5)
    #             sleep(0.5)
    #             pass

    #         prev_x = x_pos
    #         prev_y = y_pos
        
    #     a = 1
    for tile_info, group in df.groupby(by=['type', 'sub_type', 'tile_page', 'tile_row', 'tile_col']):
        pyautogui.click(MENU_DICT['road'])
        if tile_info[0] == 'road' and tile_info[2] > -1:
            road_sub_type = ROAD_TYPE_DICT[tile_info[1]]
            tile_page_point = TILE_PAGE_TO_ARROW_DICT[tile_info[2]]
            tile_pos_x = TILE_COL_DICT[tile_info[4]]
            tile_pos_y = TILE_ROW_DICT[tile_info[3]]
            pyautogui.click(road_sub_type, interval=0.25)
            pyautogui.click(tile_page_point, interval=0.25)
            pyautogui.click(x=tile_pos_x, y=tile_pos_y, interval=0.25)
            for _, row in group.iterrows():
                x_pos = int(row.x * SQUARE_SIZE_X + UPPER_LEFT_SQUARE.x)
                y_pos = int(LOWER_RIGHT_SQUARE.y - row.y * SQUARE_SIZE_Y)
                pyautogui.click(x=x_pos, y=y_pos)

def set_ground(df, map_df):
    for group_info, group in df.groupby(by=['menu', 'cat1', 'cat2', 'direction']):
        if group_info[0] not in MENU_DICT:
            continue
        # if group_info[0] != 'Road':
        #     continue
        # ground_menu = MENU_DICT[group_info[0]]
        # if group_info[0] == 'Ground 2':
        #     ground_type = GROUND_2_DICT[group_info[1]]
        # if group_info[0] == 'Ground 3':
        #     ground_type = GROUND_3_DICT[group_info[1]]
        # elif group_info[0] == 'Foliage':
        #     ground_type = FOLIAGE_DICT[group_info[1]]

        if group_info[0] not in MENU_DICT or group_info[1] not in MENU_DICT:
            continue
        pyautogui.click(MENU_DICT[group_info[0]])
        pyautogui.click(MENU_DICT[group_info[1]])

        if group_info[2] in MENU_DICT:
            pyautogui.click(MENU_DICT[group_info[2]])
        if group_info[3] in MENU_DICT:
            pyautogui.click(MENU_DICT[group_info[3]])

        for row_idx, row in group.iterrows():
            x_pos = int(row.x * SQUARE_SIZE_X + UPPER_LEFT_SQUARE.x + SQUARE_SIZE_X / 4)
            y_pos = int(LOWER_RIGHT_SQUARE.y - SQUARE_SIZE_Y / 4 - row.y * SQUARE_SIZE_Y)
            pyautogui.click(x=x_pos, y=y_pos)

            map_df.loc[row_idx, 'done'] = 1


    
def start_editor(filepath, countdown):
    if os.path.exists(args.input + '.checkpoint') and os.path.exists(args.input + '.meta.checkpoint'):
        map_df = pandas.read_csv(args.input + '.checkpoint')
        meta_df = pandas.read_csv(args.input + '.meta.checkpoint')
        start_i_page_x = meta_df['start_i_page_x'][0]
        start_i_page_y = meta_df['start_i_page_y'][0]
        prev_n_x = meta_df['prev_n_x'][0]
        prev_n_y = meta_df['prev_n_y'][0]

    else:
        map_df = pandas.read_csv(args.input)
        map_df.z = map_df.z.round().astype(int)

        if 'done' not in map_df:
            map_df['done'] = 0

        start_i_page_x = 0
        start_i_page_y = 0
        prev_n_x = START_N_SQUARES_X
        prev_n_y = START_N_SQUARES_Y

    try:    
        map_df = map_df[map_df['done'] == 0]
        # map_df.rolling(3, on='x').apply(rolling_test, kwargs={'y': map_df.y})

        # x = np.array(map_df.x.values, dtype=int)
        # y = np.array(map_df.y.values, dtype=int)
        # z = map_df.z.values

        # grid = np.full((x.max() + 1, y.max() + 1), -1)
        # grid[x, y] = z
        total_n_squares_x = int(map_df.x.max()) + 1
        total_n_squares_y = int(map_df.y.max()) + 1

        n_pages_x, n_x_remain = np.divmod(total_n_squares_x, PAGE_N_SQUARES_X, dtype=int)
        n_pages_y, n_y_remain = np.divmod(total_n_squares_y, PAGE_N_SQUARES_Y, dtype=int)
        n_x_remain = (np.floor(n_x_remain / 2) * 2).astype(int)
        n_y_remain = (np.floor(n_y_remain / 2) * 2).astype(int)

        # grid = grid[0:(n_pages_x * PAGE_N_SQUARES_X + n_x_remain), 0:(n_pages_y * PAGE_N_SQUARES_Y + n_y_remain)]

        map_df = map_df[(map_df.x >= 0) & (map_df.y >= 0) & (map_df.x < (n_pages_x * PAGE_N_SQUARES_X + n_x_remain)) & (map_df.y < (n_pages_y * PAGE_N_SQUARES_Y + n_y_remain))]

        # map_df = map_df[map_df.x.between(0, (n_pages_x * PAGE_N_SQUARES_X + n_x_remain), inclusive='left')]
        # map_df = map_df[map_df.y.between(0, (n_pages_y * PAGE_N_SQUARES_Y + n_y_remain), inclusive='left')]
                        
        total_n_squares_x = n_pages_x * PAGE_N_SQUARES_X + n_x_remain
        total_n_squares_y = n_pages_y * PAGE_N_SQUARES_Y + n_y_remain

        height = START_HEIGHT

        pyautogui.countdown(countdown)

        for i_page_y in range(n_pages_y + 1):
            for i_page_x in range(n_pages_x + 1):
                # if i_page_x > 3 or i_page_y > 3:
                #     continue

                # xmax = total_n_squares_x - i_page_x * PAGE_N_SQUARES_X
                # ymax = (i_page_y + 1) * PAGE_N_SQUARES_Y
                # if i_page_x < n_pages_x:
                #     n_squares_x = (i_page_x + 1) * PAGE_N_SQUARES_X
                #     xmin = xmax - PAGE_N_SQUARES_X
                # else:
                #     n_squares_x = i_page_x * PAGE_N_SQUARES_X + n_x_remain 
                #     xmin = xmax - n_x_remain
                # if i_page_y < n_pages_y:
                #     n_squares_y = (i_page_y + 1) * PAGE_N_SQUARES_Y
                #     ymin = ymax - PAGE_N_SQUARES_Y
                # else:
                #     n_squares_y = i_page_y * PAGE_N_SQUARES_Y + n_y_remain
                #     ymin = ymax - PAGE_N_SQUARES_Y


                if i_page_x < n_pages_x:
                    n_squares_x = (i_page_x + 1) * PAGE_N_SQUARES_X
                    xmax = total_n_squares_x - i_page_x * PAGE_N_SQUARES_X
                    xmin = xmax - PAGE_N_SQUARES_X
                    origin_x = total_n_squares_x - (i_page_x + 1) * PAGE_N_SQUARES_X
                else:
                    n_squares_x = i_page_x * PAGE_N_SQUARES_X + n_x_remain 
                    xmax = n_x_remain
                    xmin = 0
                    origin_x = 0
                if i_page_y < n_pages_y:
                    n_squares_y = (i_page_y + 1) * PAGE_N_SQUARES_Y
                    ymax = (i_page_y + 1) * PAGE_N_SQUARES_Y
                    ymin = ymax - PAGE_N_SQUARES_Y
                    origin_y = i_page_y * PAGE_N_SQUARES_Y
                else:
                    n_squares_y = i_page_y * PAGE_N_SQUARES_Y + n_y_remain
                    ymax = total_n_squares_y
                    ymin = total_n_squares_y - n_y_remain
                    origin_y = total_n_squares_y - PAGE_N_SQUARES_Y

                if xmax == xmin or ymax == ymin:
                    continue
                # if i_page_x == 0 and i_page_y == 0:
                #     continue

                if prev_n_x == START_N_SQUARES_X and prev_n_y == START_N_SQUARES_Y:
                    init = True
                else:
                    init = False
                set_n_squares(prev_n_x, prev_n_y, n_squares_x, n_squares_y, init)
                prev_n_x = n_squares_x
                prev_n_y = n_squares_y

                # sub_grid = grid[xmin:xmax, ymin:ymax]

                # height = process_segment(sub_grid, height)
                sub_df = map_df[map_df.x.between(xmin, xmax, inclusive='left') & map_df.y.between(ymin, ymax, inclusive='left')].copy(deep=True)
                sub_df.x = sub_df.x - origin_x
                sub_df.y = sub_df.y - origin_y
                height = process_segment(sub_df, height)
                # set_roads(sub_df)
                if 'menu' in map_df.columns:
                    set_ground(sub_df, map_df)

    except pyautogui.FailSafeException:
        pass
        # map_df.to_csv(args.input + '.checkpoint')
        # meta_df = pandas.DataFrame({'prev_n_x': [prev_n_x], 'prev_n_y': [prev_n_y], 'start_i_page_x': [i_page_x], 'start_i_page_y': [i_page_y]})
        # meta_df.to_csv(args.input + '.meta.checkpoint')

    pyautogui.alert(text='CMAutoEditor has finished processing the input data.', title='CMAutoEditor')
        
if __name__ == '__main__':
    sg.theme('Dark')
    sg.theme_button_color('#002366')

    #Run the gui if no arguments are inputted
    if len(sys.argv) == 1:
        display_gui()
    else:
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument('-i', '--input', required=True, help='File containing input data in csv-Format. Data is coded in x, y and z columns.')
        arg_parser.add_argument('-c', '--countdown', required=False, type=int, help='Countdown until CMAutoEditor starts clicking in CM.', default=5)
        args = arg_parser.parse_args()
    
        return_val = sg.popup_ok_cancel('CMAutoEditor is about to run on {}.'.format(args.input),
        'If you haven\'t done so yet, open up the CM Scenario Editor, go to map->Elevation and click \'Direct\'. Make sure the size is 320m x 320m.',
        'Once you are ready to start click \'Ok\'. You will then have {}s to switch back to the CM Scenario Editor.'.format(args.countdown),
        'In case something goes wrong, move the mouse cursor to one of the screen corners. This will stop CMAutoEditor.', 
        title='CMAutoEditor')
        
        if return_val == 'Cancel' or return_val is None:
            exit()
        
        start_editor(args.input, args.countdown)
    




