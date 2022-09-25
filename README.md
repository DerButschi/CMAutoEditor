# CMAutoEditor
Combat Mission is a series of wargames developed by [Battlefront](https://www.battlefront.com). The series ships with a scenario editor which, among other 
things allows the scenario designer to set an elevation value for each sqaure the map is made of. While it is possible to draw contour lines and let the editor 
interpolate between these lines it is in principle possible to set an individual value for each square. Especially for larger maps it is not really feasible 
to do this manually.

This script automates setting elevation values in the editor. It reads data from a .csv-File and "does the clicking" for the scenario designer.

## Setup
You need a working python3 (tested with python 3.9) environment. After cloning this repository do `pip install -r requirements.txt` or install 
the packages listed in the file manually.

## A Note of Caution
The script uses [PyAutoGUI](https://pyautogui.readthedocs.io/en/latest/) to do all the clicking for you. It does not which application is currently running. 
If you fail to go back to the editor during the countdown, the clicking and keyboard activations might end up in some other application currently open and do 
unintended things there. PyAutoGUI has a fail safe: Just slam the mouse cursor into on of the screen corners and the script will stop.

## How to run the script

- Set your keyboard layout to English (see Known Issues).
- Open the scenario editor and go to map->elevation. Click 'Direct' to enable direct-set mode.
- Run the script: `python cmautoeditor.py -i /path/to/data-file.csv
- You will see a countdown start to tick down.
- During that time go back to the scenario editor.
- Watch the script clicking...

## Input Data
The script takes data in csv-format with a header x,y,z. x and y denote the position of a value on the map *in units of map squares*. x=2, y=1 denotes the 3rd 
square in horizontal direction and the 2nd in vertical direction - x=0, y=0 is the origin. z values are given in meters. You can provide z-values < 0. These 
will be ignored when actually setting the elevation values but will be taken into account when calculating how large the map is.

## Screen Resolutions other than 1920x1080
I only tested the script with a screen resolution of 1920x1080. I guess for different resolution some or all GUI elements will be at different positions and/or have different sizes.
You can adapt the script to your screen resolution by modifying the parameters at the beginning of the script. You can get the correct values e.g. by making a screenshot and looking up the positions in the image manipulation programm of your choice.

```
UPPER_LEFT_SQUARE = pyautogui.Point(234,52) # center of the upper left terrain square
UPPER_RIGHT_SQUARE = pyautogui.Point(1882,52) # center of the upper right terrain square
LOWER_LEFT_SQUARE = pyautogui.Point(234,996) # center of the lower left terrain square
LOWER_RIGHT_SQUARE = pyautogui.Point(1882,996) # center of the lower right terrain square

SQUARE_SIZE_X = 16 # terrain sqaure size in horizontal direction
SQUARE_SIZE_Y = 16 # terrain sqaure size in vertical direction

START_N_SQUARES_X = 40 # default number of squares in horizontal direction
START_N_SQUARES_Y = 40 # default number of squares in vertical direction

PAGE_N_SQUARES_X = 104 # max. number of squares on screen in horizontal direction
PAGE_N_SQUARES_Y = 60 # max. number of squares on screen in vertical direction

START_HEIGHT = 20 # height value the editor starts with

POS_HORIZONTAL_PLUS = pyautogui.Point(764, 10) # position of the left plus button for horizontal size
POS_HORIZONTAL_MINUS = pyautogui.Point(764, 26) # position of the left minus button for horizontal size

POS_VERTICAL_PLUS = pyautogui.Point(1014, 10) # position of the upper plus button for vertical size
POS_VERTICAL_MINUS = pyautogui.Point(903, 10) # position of the upper minus button for vertical size
``` 



## Comments & Known Issues
- So far, the script is setup to work with a screen resolution of 1920x1080. 
- During testing, the script refused to work properly with a German keyboard layout but worked fine with an English one. I did not test any other layout.
- The script is not terribly fast. That is mainly due to the fact the apparently the editor can't handle to fast clicking or key pressing. You can experiment with the interval values in the script or with the `pyautogui.PAUSE` parameter. Use the latter with caution. Setting the value to low means you won't be able to activate PyAutoGUI's fail safe!
- The resizing of the map is necessary! Mouse scrolling is unreliable, meaning the outcome is not very predictable. However, when resizing the map, the information being cut away 
doesn't get lost. In this way, resizing can be used for exact scrolling.


