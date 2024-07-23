import argparse
import xml.etree.ElementTree as ET
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--kml_file', type=str, required=True, help='download the .kml from https://starlinkinsider.com/starlink-gateway-locations/')
parser.add_argument('--output_file', type=str, required=True)

KML_FILE = parser.parse_args().kml_file
OUTPUT_FILE = parser.parse_args().output_file
REGIONS = ['Middle East', 'South America', 'North America', 'Europe', 'Africa', 'APAC']

tree = ET.parse(KML_FILE)
root = tree.getroot()

xmlns = root.tag.split('}')[0] + '}'

ground_stations = []

for folder in root.findall(f'{xmlns}Document/{xmlns}Folder'):
    if folder.find(f'{xmlns}name').text in REGIONS:
        for placemark in folder.findall(f'{xmlns}Placemark'):
            data = placemark.find(f"{xmlns}ExtendedData")
            assert data is not None

            station = {}
            station['name'] = placemark.find(f'{xmlns}name').text
            coordinates = placemark.find(f'{xmlns}Point/{xmlns}coordinates').text.strip().split(',')
            station['coordinates'] = [float(coordinates[0]), float(coordinates[1])]
            station['status'] = data.find(f"{xmlns}Data[@name='Status']/{xmlns}value").text
            try:
                station['uplink_ghz'] = float(data.find(f"{xmlns}Data[@name='Ka Uplink Ghz']/{xmlns}value").text)
            except TypeError:
                station['uplink_ghz'] = None
            try:
                station['downlink_ghz'] = float(data.find(f"{xmlns}Data[@name='Ka Downlink Ghz']/{xmlns}value").text)
            except TypeError:
                station['downlink_ghz'] = None

            ground_stations.append(station)

print("Ground station number: ", len(ground_stations))
json_obj = {'ground_stations': ground_stations}
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, 'w') as f:
    json.dump(json_obj, f, indent=4)