from time import sleep
import pyautogui
import numpy as np
import pickle

# constants:
UPPER_LEFT_SQUARE = pyautogui.Point(234,52)
UPPER_RIGHT_SQUARE = pyautogui.Point(1882,52)
LOWER_LEFT_SQUARE = pyautogui.Point(234,996)
LOWER_RIGHT_SQUARE = pyautogui.Point(1882,996)

SQUARE_SIZE_X = 16
SQUARE_SIZE_Y = 16

START_N_SQUARES_X = 40
START_N_SQUARES_Y = 40

PAGE_N_SQUARES_X = 104
PAGE_N_SQUARES_Y = 60

START_HEIGHT = 20

POS_HORIZONTAL_PLUS = pyautogui.Point(764, 10)
POS_HORIZONTAL_MINUS = pyautogui.Point(764, 26)

POS_VERTICAL_PLUS = pyautogui.Point(1014, 10)
POS_VERTICAL_MINUS = pyautogui.Point(903, 10)

pyautogui.PAUSE = 0.2

def set_height(current_height, target_height):
    if current_height == target_height:
        return
    elif current_height < target_height:
        n_diff = target_height - current_height
        # key_list = ['+'] * n_diff
        pyautogui.press('+', presses=n_diff, interval=0.5)
    else:
        n_diff = current_height - target_height
        # key_list = ['-'] * n_diff
        pyautogui.press('-', presses=n_diff, interval=0.5)

    sleep(1)

def process_segment(grid, start_height):
    print(grid.shape)
    values = np.unique(grid)
    min_height = np.min(grid)
    
    set_height(start_height, min_height)
    height = min_height

    for val in values:
        indices0, indices1 = np.where(grid == val)

        set_height(height, val)

        for i in range(len(indices1)):
            idx0 = indices0[i]
            idx1 = indices1[i]
            x_pos = int(idx0 * SQUARE_SIZE_X + UPPER_LEFT_SQUARE.x)
            y_pos = int(LOWER_RIGHT_SQUARE.y - (PAGE_N_SQUARES_Y - grid.shape[1] + idx1) * SQUARE_SIZE_Y)
            height = val

            pyautogui.click(x=x_pos, y=y_pos)

    return height

def set_n_squares(start_n_x, start_n_y, n_x, n_y):
    n_clicks_x = abs(int((start_n_x - n_x) / 2))
    n_clicks_y = abs(int((start_n_y - n_y) / 2))

    for i in range(n_clicks_x):
        if n_x <= start_n_x:
            pyautogui.click(POS_HORIZONTAL_MINUS, interval=0.5)
        else:
            pyautogui.click(POS_HORIZONTAL_PLUS, interval=0.5)
    
    for i in range(n_clicks_y):
        if n_y <= start_n_y:
            pyautogui.click(POS_VERTICAL_MINUS, interval=0.5)
        else:
            pyautogui.click(POS_VERTICAL_PLUS, interval=0.5)

if __name__ == '__main__':
    pyautogui.countdown(5)

    # load height map
    grid = pickle.load(open('heightmap.pkl', 'rb')).astype(int)

    n_pages_x, n_x_remain = np.divmod(grid.shape[0], PAGE_N_SQUARES_X)
    n_pages_y, n_y_remain = np.divmod(grid.shape[1], PAGE_N_SQUARES_Y)
    n_x_remain = (np.floor(n_x_remain / 2) * 2).astype(int)
    n_y_remain = (np.floor(n_y_remain / 2) * 2).astype(int)

    grid = grid[0:(n_pages_x * PAGE_N_SQUARES_X + n_x_remain), 0:(n_pages_y * PAGE_N_SQUARES_Y + n_y_remain)]

    height = START_HEIGHT
    prev_n_x = START_N_SQUARES_X
    prev_n_y = START_N_SQUARES_Y
    for i_page_y in range(n_pages_y + 1):
        for i_page_x in range(n_pages_x + 1):
            if i_page_x > 3 or i_page_y > 3:
                continue

            xmax = grid.shape[0] - i_page_x * PAGE_N_SQUARES_X
            ymax = (i_page_y + 1) * PAGE_N_SQUARES_Y
            if i_page_x < n_pages_x:
                n_squares_x = (i_page_x + 1) * PAGE_N_SQUARES_X
                xmin = xmax - PAGE_N_SQUARES_X
            else:
                n_squares_x = n_x_remain
                xmin = xmax - n_x_remain
            if i_page_y < n_pages_y:
                n_squares_y = (i_page_y + 1) * PAGE_N_SQUARES_Y
                ymin = ymax - PAGE_N_SQUARES_Y
            else:
                n_squares_y = i_page_y * PAGE_N_SQUARES_Y
                ymin = ymax - PAGE_N_SQUARES_Y

            set_n_squares(prev_n_x, prev_n_y, n_squares_x, n_squares_y)
            prev_n_x = n_squares_x
            prev_n_y = n_squares_y

            sub_grid = grid[xmin:xmax, ymin:ymax]

            height = process_segment(sub_grid, height)




