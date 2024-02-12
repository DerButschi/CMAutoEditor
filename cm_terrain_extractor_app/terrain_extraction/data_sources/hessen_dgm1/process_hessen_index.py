import json
import os
import requests

data = json.load(open(os.path.join("terrain_extraction", "raw_data_indices", "hessen.geojson"), encoding="utf-8"))

data_dict = {
    'type': data['type'],
    'name': data['name'],
    'crs': data['crs'],
    'features': []
}

replace_lk_dict = {
    'Wetterau': 'Wetteraukreis',
    'Main-Kinzig': 'Main-Kinzig-Kreis',
    'Vogelsberg': 'Vogelsbergkreis',
    'Fulda': 'Landkreis Fulda',
    'Groß-Gerau': 'Landkreis Groß-Gerau',
    'Offenbach': 'Landkreis Offenbach',
    'Kreisfreie Stadt Darmstadt': 'Stadt Darmstadt',
    'Bergstraße': 'Landkreis Bergstraße',
    'Darmstadt-Dieburg': 'Landkreis Darmstadt-Dieburg',
    'Kreisfreie Stadt Offenbach am Main': 'Stadt Offenbach am Main',
    'Schwalm-Eder': 'Schwalm-Eder-Kreis',
    'Hersfeld-Rotenburg': 'Landkreis Hersfeld-Rotenburg',
    'Werra-Meißner': 'Werra-Meißner-Kreis',
    'Kassel': 'Landkreis Kassel',
    'Waldeck-Frankenberg': 'Landkreis Waldeck-Frankenberg',
    'Kreisfreie Stadt Kassel': 'Stadt Kassel',
    'Hochtaunus': 'Hochtaunuskreis',
    'Kreisfreie Stadt Frankfurt am Main': 'Stadt Frankfurt am Main',
    'Rheingau-Taunus': 'Rheingau-Taunus-Kreis',
    'Landeshauptstadt Wiesbaden': 'Stadt Wiesbaden',
    'Main-Taunus': 'Main-Taunus-Kreis',
    'Limburg-Weilburg': 'Landkreis Limburg-Weilburg',
    'Marburg-Biedenkopf': 'Landkreis Marburg-Biedenkopf',
    'Lahn-Dill': 'Lahn-Dill-Kreis',
    'Gießen': 'Landkreis Gießen',
}

replace_gmde_dict = {
    'Erbach': 'Erbach (Odenwald)',
    'Höchst i. Odw.': 'Höchst i. Odw',
    'Neukirchen': 'Neukirchen (Knüllgebirge)',
    'Breitenbach a. Herzberg': 'Breitenbach am Herzberg',
    'Söhrewald': 'S%E2%94%9C%C3%82hrewald',
    'Schmitten im Taunus': 'Schmitten',
    'Bad Homburg v. d. Höhe': 'Bad Homburg v.d.Höhe',
    'Heuchelheim a. d. Lahn': 'Heuchelheim a.d. Lahn'
}

def replace_lk_name(name):
    if name in replace_lk_dict:
        return replace_lk_dict[name]
    else:
        return name

def replace_gmde_name(name):
    if name in replace_gmde_dict:
        return replace_gmde_dict[name]
    else:
        return name

gmde_with_error = []

for feature in data['features']:
    entry = {}
    entry['type'] = feature['type']
    entry['properties'] = {
        'file_name': replace_gmde_name(feature['properties']['GMDE_BZ']),
        'folder': replace_lk_name(feature['properties']['KREIS_BZ'])
        # 'url': 'https://gds.hessen.de/downloadcenter/20240127/3D-Daten/Digitales Geländemodell (DGM1)/{}/{} - DGM1.zip'.format(
        #     replace_lk_name(feature['properties']['KREIS_BZ']), 
        #     replace_gmde_name(feature['properties']['GMDE_BZ'])
        # )
    }
    # try:
    #     assert requests.head(entry['properties']['url']).status_code == 200
    # except:
    #     print(entry['properties']['url'])

    entry['geometry'] = feature['geometry']
    data_dict['features'].append(entry)

with open(os.path.join("terrain_extraction", "data_sources", "hessen_dgm1", "hessen_dgm1.geojson"), 'w', encoding="utf-8") as f:
    json.dump(data_dict, f)
