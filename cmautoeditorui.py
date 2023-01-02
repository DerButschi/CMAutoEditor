# Copyright (C) 2022  Archie Hill

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

import PySimpleGUI as sg
import cmautoeditor as cmae

if __name__ == '__main__':
    sg.theme('Dark')
    
    # Construct window layout
    layout = [
        [sg.Titlebar('CM Auto Editor')],
        [sg.Text('Select file: ')], 
        [sg.Input(), sg.FileBrowse(key='filepath', file_types=(('CSV files', '*.csv'),))],
        [sg.Text('Countdown: '), sg.InputCombo(key='countdown',values=[5, 10, 15, 20, 25, 30], default_value=10)],
        [sg.Text(text='', key='error_text')],
        [sg.Submit('Start Autoclicker', key='submit'), sg.Exit()]]

    # Create window with layout
    window = sg.Window('CM Auto Editor', layout)
    
    # Loop until window needs closing
    while True:
        # Read UI inputs
        event, values = window.read()
        
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        
        if event == 'submit':
            if values['filepath'] == '':
                window['error_text'].update('Select a file before submitting')
            else:
                break
            
    window.close()
    # Start autoclicker with UI inputs
    if values['filepath'] == '':
        cmae.start_autoclicker(values['filepath'], values['countdown'])
            

