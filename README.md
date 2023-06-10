# CMAutoEditor
Combat Mission is a series of wargames developed by [Battlefront](https://www.battlefront.com). The series ships with a scenario editor which, among other 
things allows the scenario designer to set an elevation value for each sqaure the map is made of. While it is possible to draw contour lines and let the editor 
interpolate between these lines it is in principle possible to set an individual value for each square. Especially for larger maps it is not really feasible 
to do this manually.

This repository provides a collection of scripts:
- dgm2cm.py converts DGM (Digitales Geländemodell) elevation data into a format readable by cmautoeditor.py
- geotiff2cm.py converts elevation data in geotiff format into a format readable by cmautoeditor.py
- osm2cm converts OpenStreetMap data (in geojson-format) into cmautoeditor.py readable format
- cmautoeditor.py automates "clicking in the scenario editor: It reads external elevation and terrain data and "converts" it into a Combat Mission map by using the editor just like a human scenario designer would - albeit with superhuman speed!

## Installing CMAutoEditor
There are basically two options.
### Install binaries
Download the latest Windows .exe-files here:
[CMAutoEditor-latest](https://github.com/DerButschi/CMAutoEditor/releases/latest/download/release.zip)
The files are compiled on the latest Windows version and tested on Windows 10. 

### Building from source
If the binary files don't work on your machine or you want to build from source go to the [latest release](https://github.com/DerButschi/CMAutoEditor/releases/latest) 
and under Assets click "Source code" (.zip or .tar.gz)

CMAutoEditor depends on some libraries that are only available at Conda-Forge, so you need a python environment with conda, either 
[Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://docs.conda.io/projects/conda/en/stable/glossary.html#anaconda-glossary).
Currently CMAutoEditor requires python version >= 3.10. Note: Unfortunately this makes CMAutoEditor incompatible with Windows versions below Windows 10.

Once you have a conda version installed, you can directly "clone" the enviroment from the file included with the source code:
```
conda create -n py310_cmautoeditor -c conda-forge --override-channels --file conda_requirements_py310.txt
```

When running the scripts, do:

```
conda activate py310_cmautoeditor
```
before executing the actual code.

## A Note of Caution
The script uses [PyAutoGUI](https://pyautogui.readthedocs.io/en/latest/) to do all the clicking for you. It does not which application is currently running. 
If you fail to go back to the editor during the countdown, the clicking and keyboard activations might end up in some other application currently open and do 
unintended things there. PyAutoGUI has a fail safe: Just slam the mouse cursor into on of the screen corners and the script will stop.

