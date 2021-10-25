import ee
import google.auth
from google.oauth2 import service_account
import pandas as pd
import geojson
import json

import folium
import geehydro
from datetime import datetime as dt
from IPython.display import display, HTML

import folium
from folium import plugins
from IPython.display import HTML, display
import webbrowser

import shapefile
from pyproj import Proj, transform

def load_shapefile(shapefile):
    source_ds = ogr.Open(shapefile)
    source_layer = source_ds.GetLayer()
    return source_layer

def reproject_shapefile(layer, source_crs, dest_crs):
    source = osr.SpatialReference()
    source.ImportFromEPSG(int(source_crs))
    target = osr.SpatialReference()
    target.ImportFromEPSG(int(dest_crs))
    transform = osr.CoordinateTransformation(source, target)
    reprojected_layer = layer.Transform(transform)
    return reprojected_layer

try:
    service_account= 'gee-access@sublime-bliss-262212.iam.gserviceaccount.com'
    credentials = ee.ServiceAccountCredentials(service_account, 'sublime-bliss-262212-25172b196bc4.json')
    ee.Initialize(credentials)

except:
    ee.Authenticate()
    ee.Initialize()

def retrieve_monthly_precipitation_data(first_year, last_year, studyarea, studyarea_epsg, save_maps = False, visualize = True):
    # Data source:
    # Copernicus Climate Change Service (C3S) (2017): ERA5: Fifth generation of ECMWF atmospheric reanalyses of the
    # global climate. Copernicus Climate Change Service Climate Data Store (CDS), (date of access),
    # https://cds.climate.copernicus.eu/cdsapp#!/home

    #climate_data = ee.ImageCollection("ECMWF/ERA5/MONTHLY")
    # yearly_precipitation = climate_data.select('total_precipitation').filterDate(startDate, endDate).reduce(ee.Reducer.sum())
    climate_data = ee.Image('WORLDCLIM/V1/BIO')
    #yearly_precipitation = climate_data.select('bio12').filterDate(startDate, endDate).reduce(ee.Reducer.sum())
    yearly_precipitation = climate_data.select('bio12')
    yearly_precipitation_variance = climate_data.select('bio15')

    #clip by extent
    features = []

    with shapefile.Reader(studyarea) as shp:
        geojson_features = shp.__geo_interface__

    xcoords = []
    ycoords = []
    for f in geojson_features['features']:
        geom = f['geometry']
        for coord in geom['coordinates']:
            if type(coord) == float:  # then its a point feature
                xcoords.append(geom['coordinates'][0])
                ycoords.append(geom['coordinates'][1])
            elif type(coord) == list:
                for c in coord:
                    ycoords.append(c[0])
                    xcoords.append(c[1])

    from pyproj import Proj, transform
    inProj = Proj(studyarea_epsg)
    outProj = Proj('EPSG:4326')

    ycoords, xcoords = transform(inProj, outProj, ycoords,xcoords)

    extent = [
        [min(xcoords), min(ycoords)],
        [max(xcoords), min(ycoords)],
        [max(xcoords), max(ycoords)],
        [min(xcoords), max(ycoords)]
    ]

    polygon_extent = ee.Geometry.Polygon(extent)

    yearly_precipitation_clipped = yearly_precipitation.clip(polygon_extent)
    yearly_precipitation_variance_clipped = yearly_precipitation_variance.clip(polygon_extent)

    mean_dic = yearly_precipitation_clipped.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=polygon_extent,
        scale=30,
        maxPixels=1e9
    )
    print("mean annual precipitation: " + str(mean_dic.getInfo()))

    if save_maps is True:
        def get_url(name, image):
            path = image.getDownloadURL({
                'name': name
            })

            webbrowser.open_new_tab(path)
            return path

        url_nithi = get_url('mean_annual_precipitation', yearly_precipitation_clipped)
        url_nithi = get_url('variance_annual_precipitation', yearly_precipitation_variance_clipped)



    if visualize is True:

        basemaps = {
            'Google Maps': folium.TileLayer(
                tiles='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
                attr='Google',
                name='Google Maps',
                overlay=True,
                control=True
            ),
            'Google Satellite': folium.TileLayer(
                tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                attr='Google',
                name='Google Satellite',
                overlay=True,
                control=True
            ),
            'Google Terrain': folium.TileLayer(
                tiles='https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}',
                attr='Google',
                name='Google Terrain',
                overlay=True,
                control=True
            ),
            'Google Satellite Hybrid': folium.TileLayer(
                tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
                attr='Google',
                name='Google Satellite',
                overlay=True,
                control=True
            ),
            'Esri Satellite': folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri',
                name='Esri Satellite',
                overlay=True,
                control=True
            )
        }

        def add_ee_layer(self, ee_object, vis_params, name):

            try:
                # display ee.Image()
                if isinstance(ee_object, ee.image.Image):
                    map_id_dict = ee.Image(ee_object).getMapId(vis_params)
                    folium.raster_layers.TileLayer(
                        tiles=map_id_dict['tile_fetcher'].url_format,
                        attr='Google Earth Engine',
                        name=name,
                        overlay=True,
                        control=True
                    ).add_to(self)
                # display ee.ImageCollection()
                elif isinstance(ee_object, ee.imagecollection.ImageCollection):
                    ee_object_new = ee_object.mosaic()
                    map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
                    folium.raster_layers.TileLayer(
                        tiles=map_id_dict['tile_fetcher'].url_format,
                        attr='Google Earth Engine',
                        name=name,
                        overlay=True,
                        control=True
                    ).add_to(self)
                # display ee.Geometry()
                elif isinstance(ee_object, ee.geometry.Geometry):
                    folium.GeoJson(
                        data=ee_object.getInfo(),
                        name=name,
                        overlay=True,
                        control=True
                    ).add_to(self)
                # display ee.FeatureCollection()
                elif isinstance(ee_object, ee.featurecollection.FeatureCollection):
                    ee_object_new = ee.Image().paint(ee_object, 0, 2)
                    map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
                    folium.raster_layers.TileLayer(
                        tiles=map_id_dict['tile_fetcher'].url_format,
                        attr='Google Earth Engine',
                        name=name,
                        overlay=True,
                        control=True
                    ).add_to(self)

            except:
                print("Could not display {}".format(name))

        # Add EE drawing method to folium.
        folium.Map.add_ee_layer = add_ee_layer

        vis_params = {
            'min': 0,
            'max': 2000,
            'palette': ['006633', 'E5FFCC', '662A00', 'D8D8D8', 'F5F5F5']}

        vis_params2 = {
            'min': 90,
            'max': 100,
            'palette': ['006633', 'E5FFCC', '662A00', 'D8D8D8', 'F5F5F5']}

        # Create a folium map object.
        my_map = folium.Map(location=[20, 0], zoom_start=3, height=500)

        # Add custom basemaps
        basemaps['Google Maps'].add_to(my_map)
        basemaps['Google Satellite Hybrid'].add_to(my_map)

        my_map.add_ee_layer(yearly_precipitation_clipped, vis_params, 'mean_annual_precipitation')
        my_map.add_ee_layer(yearly_precipitation_variance_clipped, vis_params2, 'variance_annual_precipitation')

        my_map.save("map.html")

        webbrowser.open("map.html")

