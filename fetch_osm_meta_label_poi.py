import numpy as np
import requests
import pandas as pd
import logging
import os
import time
import datetime

logging.basicConfig(filename='label_downloader.log', filemode='w')


def lldistkmv(latlon_list1, latlon_list2=None):
    if not latlon_list2:
        latlon_list2 = latlon_list1 + 1

    L1, A1 = latlon_list1.shape
    L2, A2 = latlon_list2.shape

    radius = 6378.137
    latv1 = np.dot(latlon_list1[:, 0:1], (np.pi / 180) * np.ones((1, L2)))
    latv2 = np.dot(np.ones((L1, 1)), latlon_list2[:, 0:1].T * (np.pi / 180))
    lonv1 = np.dot(latlon_list1[:, 1:2], (np.pi / 180) * np.ones((1, L2)))
    lonv2 = np.dot(np.ones((L1, 1)), latlon_list2[:, 1:2].T * (np.pi / 180))

    deltaLat = latv2 - latv1
    deltaLon = lonv2 - lonv1
    a = np.sin(deltaLat / 2) ** 2 + np.cos(latv1) * np.cos(latv2) * np.sin(deltaLon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d1km = radius * c  # Haversine distance

    x = deltaLon * np.cos((latv1 + latv2) / 2)
    y = deltaLat.copy()
    d2km = radius * np.sqrt(x ** 2 + y ** 2)  # Pythagoran distance

    return d1km, d2km, y * radius, x * radius


def poi_count(poi_names, poi_dict):
    poi_count_mat = np.zeros((len(poi_dict), len(poi_names)), dtype=int)
    key2index = {x: i for i, x in enumerate(poi_names)}
    for i, item in enumerate(poi_dict):
        for key in item:
            poi_count_mat[i, key2index[key]] += item[key]
    return poi_count_mat


def prepare_map_metadata_label(df, chunk_num,
                               meta_folder_path, label_folder_path, poi_folder_path):
    """
    Download the metadata from the server and parse them in a dataframe
    :param df:
    :return:
    """
    # overpass_url = "https://lz4.overpass-api.de/api/interpreter"
    overpass_url = "http://localhost:12345/api/interpreter"

    cols = ['mesh', 'buildings', 'highway', 'peak', 'water', 'river', 'railway', 'railstation', 'park', 'playground',
            'roads', 'airport', 'trail', 'farmland', 'grassland']
    metadata = pd.DataFrame(columns=cols)

    labels = pd.DataFrame(
        columns=['mesh', 'buildings_less', 'buildings_some', 'buildings_more', 'highway', 'peak', 'water', 'river',
                 'railway', 'railstation', 'park', 'playground', 'roads_less', 'road_some', 'road_more', 'airport',
                 'trail', 'farmland', 'grassland'])

    # poi
    poi_amenity_dict, poi_amenity_names = list(), set()

    for idx, row in df.iterrows():
        lat, lon = row['lat'], row['lon']
        lat = round(float(lat), 5)
        lon = round(float(lon), 5)
        # lat_lon = str(lat) + '_' + str(lon)
        mesh = row['meshcode']
        _, _, data1, data2 = lldistkmv(np.array([[lat, lon]]))
        lat_diff = 0.245 / data1[0, 0]
        lon_diff = 0.245 / data2[0, 0]

        lat1 = round(lat + lat_diff, 5)
        lat2 = round(lat - lat_diff, 5)
        lon1 = round(lon + lon_diff, 5)
        lon2 = round(lon - lon_diff, 5)

        overpass_query = """
        [out:json];
        (
          way({}, {}, {}, {});
          node({}, {}, {}, {});
          relation({}, {}, {}, {});
          );
        out body;
        """.format(lat2, lon2, lat1, lon1, lat2, lon2, lat1, lon1, lat2, lon2, lat1, lon1)

        r = requests.get(overpass_url,
                         params={'data': overpass_query})

        if r.status_code == 200:
            data = r.json()

            buildings = 0
            highway = 0
            peak = 0
            water = 0
            river = 0
            railway = 0
            railstation = 0
            park = 0
            playground = 0
            roads = 0
            airport = 0
            trail = 0
            farmland = 0
            grassland = 0

            # POI
            poi_amenity_results_i, poi_amenity_names_i = list(), set()
            poi_amenity, poi_all = dict(), dict()

            for i in range(len(data['elements'])):

                # poi
                if 'tags' in data['elements'][i]:
                    if 'amenity' in data['elements'][i]['tags']:
                        poi_amenity[data['elements'][i]['tags']['amenity']] = poi_amenity.get(
                            data['elements'][i]['tags']['amenity'], 0) + 1



                # metadata
                if 'tags' in data['elements'][i].keys():
                    tag = data['elements'][i]['tags']
                    keys = tag.keys()
                    if 'leisure' in keys:
                        if tag['leisure'] == 'park':
                            park += 1
                        if tag['leisure'] == 'playground':
                            playground += 1
                    elif 'highway' in keys:
                        if tag['highway'] == 'path':
                            trail += 1
                        elif tag['highway'] == 'trunk':
                            highway += 1
                        else:
                            roads += 1
                    elif 'building' in keys:
                        buildings += 1
                    elif 'aeroway' in keys:
                        airport += 1
                    elif 'waterway' in keys:
                        river += 1
                    elif 'natural' in keys:
                        if tag['natural'] == 'peak':
                            peak += 1
                        elif tag['natural'] == 'water':
                            water += 1
                        elif tag['natural'] == 'grassland':
                            grassland += 1
                    elif 'landuse' in keys:
                        if tag['landuse'] == 'farmland':
                            farmland += 1
                    elif 'railway' in keys:
                        if tag['railway'] == 'station':
                            railstation += 1
                        elif tag['railway'] == 'tram' or tag['railway'] == 'rail':
                            railway += 1
                    elif 'public_transport' in keys:
                        if tag['public_transport'] == 'station':
                            railstation += 1

            poi_amenity_results_i.append(poi_amenity)
            poi_amenity_names_i |= set(poi_amenity.keys())
            poi_amenity_dict.extend(poi_amenity_results_i)
            poi_amenity_names |= poi_amenity_names_i

            metadata.loc[len(metadata)] = [mesh, buildings, highway, peak, water, river, railway, railstation, park,
                                           playground, roads, airport, trail, farmland, grassland]
            labels.loc[len(labels)] = [mesh,
                                       int(bool(0 < buildings <= 3)),
                                       int(bool(3 < buildings <= 60)),
                                       int(bool(buildings > 60)),
                                       int(bool(highway)),
                                       int(bool(peak)),
                                       int(bool(water)),
                                       int(bool(river)),
                                       int(bool(railway)),
                                       int(bool(railstation)),
                                       int(bool(park)),
                                       int(bool(playground)),
                                       int(bool(0 < roads <= 15)),
                                       int(bool(15 < roads <= 30)),
                                       int(bool(roads > 30)),
                                       int(bool(airport)),
                                       int(bool(trail)),
                                       int(bool(farmland)),
                                       int(bool(grassland))]

        else:
            print('Failed to fetch metadata for {}'.format(mesh))
            logging.warning('code: {}, reason: {}, lat_lon: {}'.format(r.status_code, r.reason, mesh))

    metadata.to_csv(os.path.join(meta_folder_path, 'mesh_metadata_{}.csv'.format(chunk_num)), index=False)
    labels.to_csv(os.path.join(label_folder_path, 'mesh_label_{}.csv'.format(chunk_num)), index=False)

    # save poi
    poi_amenity_names = list(poi_amenity_names)
    poi_amenity_count = poi_count(poi_amenity_names, poi_amenity_dict)
    amenity = pd.DataFrame()
    amenity['mesh'] = list(df['meshcode'].values)
    amenity['poi_count'] = poi_amenity_count.tolist()

    amenity.to_csv(os.path.join(poi_folder_path, 'mesh_poi_{:d}.csv'.format(chunk_num)),
                   index=None, header=True)
    with open(os.path.join(poi_folder_path, 'mesh_amenity_names_{:d}.txt'.format(chunk_num)), mode='wt',
              encoding='utf-8') as f:
        for word in poi_amenity_names:
            f.write(word)
            f.write('\n')


if __name__ == '__main__':
    df = pd.read_csv('./data/example.csv')
    print('processing {} of locations'.format(len(df)))

    chunk_size = 1280
    chunk_number = len(df) // chunk_size + 1


    # create the following dirctories to save the data
    meta_folder_path = './data/metadata'
    label_folder_path = './data/labels'
    poi_folder_path = './data/pois'

    for cn in range(chunk_number):
        start_time = time.time()
        start = cn * chunk_size
        end = min(start + chunk_size, len(df))
        temp_df = df[start:end]
        prepare_map_metadata_label(temp_df, cn, meta_folder_path, label_folder_path, poi_folder_path)

        end_time = time.time()
        t = str(datetime.timedelta(seconds=round((end_time - start_time), 0)))
        t_left = str(datetime.timedelta(seconds=round((end_time - start_time) * (chunk_number - cn - 1), 0)))
        print('Chunk {:d}/{:d} is complete: {}, estimate {} left to finish'.format(cn + 1,
                                                                                   chunk_number,
                                                                                   t,
                                                                                   t_left))
