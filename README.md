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

## Known Issues
- So far, the script is setup to work with a screen resolution of 1920x1080. 
- During testing, the script refused to work properly with a German keyboard layout but worked fine with an English one. I did not test any other layout.

