import time
import subprocess
import shutil
import tempfile

import os
import sys
import binascii
from qgis.core import (QgsProcessing,
                       QgsProcessingAlgorithm,
                       QgsProcessingException,
                       QgsProcessingOutputNumber,
                       QgsProcessingParameterDistance,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterVectorDestination,
                       QgsProcessingParameterRasterDestination,
                       QgsExpressionContext,
                       QgsProcessingUtils)
from qgis import processing
import gdal

outputdir = r"C:\Users\morit\OneDrive - Universität Münster\PhD\Kooperation_GIZ\Data\Optimization_Enerata\bash_test"
flow_barrier = r"C:\Users\morit\OneDrive - Universität Münster\PhD\Kooperation_GIZ\Data\Optimization_Enerata\Output Enerata\tests\optimization_test\dem_1.tif"
processing.run("grass7:r.flow", {
        'elevation': dem,
        'aspect': None,
        'barrier': flow_barrier,
        'skip': None, 'bound': None, '-u': False, '-3': False, '-m': False, 'flowline': 'TEMPORARY_OUTPUT',
        'flowlength': os.path.join(outputdir, 'flow_path_length.tif'),
        'flowaccumulation': 'TEMPORARY_OUTPUT', 'GRASS_REGION_PARAMETER': None, 'GRASS_REGION_CELLSIZE_PARAMETER': 0,
        'GRASS_RASTER_FORMAT_OPT': '', 'GRASS_RASTER_FORMAT_META': '', 'GRASS_OUTPUT_TYPE_PARAMETER': 0,
        'GRASS_VECTOR_DSCO': '', 'GRASS_VECTOR_LCO': '', 'GRASS_VECTOR_EXPORT_NOCAT': False})
