import subprocess
import os
import sys
import tempfile
import binascii
import time
import shutil
import numpy as np
import math
import pandas as pd
def setup_grass(grass7bin, myepsg):
    # grass7bin = 'grass78'

    # # uncomment when using standalone WinGRASS installer
    # # grass7bin = r'C:\Program Files (x86)\GRASS GIS 7.9.0\grass79.bat'
    # # this can be avoided if GRASS executable is added to PATH
    startcmd = [grass7bin, '--config', 'path']
    try:
        p = subprocess.Popen(startcmd, shell=False,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
    except OSError as error:
        sys.exit("ERROR: Cannot find GRASS GIS start script"
                 " {cmd}: {error}".format(cmd=startcmd[0], error=error))

    if p.returncode != 0:
        sys.exit(
            "ERROR: Issues running GRASS GIS start script" " {cmd}: {error}".format(cmd=' '.join(startcmd), error=err))
    gisbase = out.decode('utf8').strip(os.linesep)

    # set GISBASE environment variable
    os.environ['GISBASE'] = gisbase
    #
    # define GRASS-Python environment
    grass_pydir = os.path.join(gisbase, "etc", "python")
    sys.path.append(grass_pydir)

    # define GRASS-Python environment
    gpydir = os.path.join(gisbase, "etc", "python")
    sys.path.append(gpydir)
    if sys.platform.startswith('win'):
        gisdb = os.path.join(os.getenv('APPDATA', 'grassdata'))
    gisdb = os.path.join(tempfile.gettempdir(), 'grassdata')
    try:
        os.stat(gisdb)
    except:
        os.mkdir(gisdb)
    string_length = 16
    location = binascii.hexlify(os.urandom(string_length)).decode('ascii')
    mapset = 'PERMANENT'
    location_path = os.path.join(gisdb, location)
    startcmd = grass7bin + ' -c epsg:' + myepsg + ' -e ' + location_path

    print(startcmd)
    p = subprocess.Popen(startcmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode != 0:
        print(sys.stderr, 'ERROR: %s' % err)
        print(sys.stderr, 'ERROR: Cannot generate location (%s)' % startcmd)
        sys.exit(-1)
    else:
        print('Created location %s' % location_path)

    # Now the location with PERMANENT mapset exists.

    ########
    # Now we can use PyGRASS or GRASS Scripting library etc. after
    # having started the session with gsetup.init() etc

    # Set GISDBASE environment variable
    os.environ['GISDBASE'] = gisdb
    path = os.getenv('LD_LIBRARY_PATH')
    dir = os.path.join(gisbase, 'lib')
    if path:
        path = dir + os.pathsep + path
    else:
        path = dir
    os.environ['LD_LIBRARY_PATH'] = path

    # language
    os.environ['LANG'] = 'en_US'
    os.environ['LOCALE'] = 'C'

    path = os.getenv('PYTHONPATH')
    dirr = os.path.join(gisbase, 'etc', 'python')
    if path:
        path = dirr + os.pathsep + path
    else:
        path = dirr
    os.environ['PYTHONPATH'] = path

    # import grass python libraries
    import grass.script as gscript
    import grass.script.setup as gsetup
    rcfile = gsetup.init(gisbase, gisdb, location, mapset)
    gscript.message('Current GRASS GIS 7 environment:')
    print(gscript.gisenv())
    return gscript.gisenv()

def calculate_rusle(solutions,input_dem, input_landuse_raster, input_watersheds, input_contour_lines, input_slope_degrees,input_r_factor, input_k_factor, input_c_factor, savemaps = False, outputdir = None):
    gis_env = setup_grass(grass7bin = r'C:\OSGeo4W64\bin\grass78.bat', myepsg = '32637')
    def compute_rusle_per_solution(solution, processes, outputdirectories):
        try:
            os.mkdir(os.path.join(gis_env['GISDBASE'], str(gis_env['LOCATION_NAME']),str(solution._solution_id)))
        except:
            shutil.rmtree(os.path.join(gis_env['GISDBASE'], str(gis_env['LOCATION_NAME']),str(solution._solution_id)), ignore_errors=False, onerror=None)
            os.mkdir(os.path.join(gis_env['GISDBASE'], str(gis_env['LOCATION_NAME']), str(solution._solution_id)))
        shutil.copytree(os.path.join(gis_env['GISDBASE'], str(gis_env['LOCATION_NAME']),"PERMANENT"), os.path.join(gis_env['GISDBASE'], str(gis_env['LOCATION_NAME']),str(solution._solution_id),"PERMANENT"))
        bash_location = os.path.join(gis_env['GISDBASE'], str(gis_env['LOCATION_NAME']),str(solution._solution_id))
        outputdir = os.path.join(gis_env['GISDBASE'], str(gis_env['LOCATION_NAME']),str(solution._solution_id), "PERMANENT")
        outputdirectories.append(os.path.join(gis_env['GISDBASE'], str(gis_env['LOCATION_NAME']),str(solution._solution_id)))
        #1. select barriers from selected spatial units (sub-watersheds)
        indizes_protected_areas = tuple(np.where(solution.representation == True)[0])
        extent =  '357194.6588,364958.1745,1145261.853,1156020.7409 [EPSG:32637]'

        # names used in GRASS GIS functions
        dem_ras = "demraster"
        landuse_ras = "landuse_ras"
        rfactor_ras = "rfactor_ras"
        kfactor_ras = "kfactor_ras"
        cfactor_ras = "cfactor_ras"
        slope_degrees = "slope_degrees_ras"
        output_flowline_ras = "flowlineraster"
        output_flowlength_ras = "flowlengthraster"
        output_flowacc_ras = "flowaccraster"
        protected_watersheds_ras = "protected_watersheds_ras"
        selected_contour_lines_ras = "selected_contour_lines_ras"
        selected_contour_lines_shp = "selected_contour_lines_shp"
        watersheds_shp = "watersheds_shp"
        selected_watersheds_shp = "selected_watersheds_shp"
        m_ras = "m_ras"
        s_factor_ras = "sfactor_ras"
        p_factor_ras = "pfactor_ras"
        lfactor_ras = "lfactor_ras"
        lsfactor_ras = "lsfactor_ras"
        rusle_ras = "rusle_ras"
        csv_stats = os.path.join(gis_env['GISDBASE'], str(gis_env['LOCATION_NAME']),str(solution._solution_id),"rusle_total.csv")
        pi_constant = math.pi
        with open(os.path.join(bash_location,"rusle.sh"), "w") as f:
            f.write("#!/bin/bash" + "\n")
            f.write('g.proj -c epsg=32637' + "\n")
            f.write(r'r.in.gdal input={} band=1 output={} --overwrite -o'.format(input_dem, dem_ras) + "\n")
            f.write(r'r.in.gdal input={} band=1 output={} --overwrite -o'.format(input_landuse_raster,
                                                                                 landuse_ras) + "\n")
            f.write(r'r.in.gdal input={} band=1 output={} --overwrite -o'.format(input_r_factor,
                                                                                 rfactor_ras) + "\n")
            f.write(r'r.in.gdal input={} band=1 output={} --overwrite -o'.format(input_k_factor,
                                                                                 kfactor_ras) + "\n")
            f.write(r'r.in.gdal input={} band=1 output={} --overwrite -o'.format(input_c_factor,
                                                                                 cfactor_ras) + "\n")
            f.write(r'r.in.gdal input={} band=1 output={} --overwrite -o'.format(input_slope_degrees,
                                                                                 slope_degrees) + "\n")

            f.write(
                r'v.in.ogr min_area=0.0 snap=-1.0 input={} layer="contour_lines_with_watershed_id_enerata" output="contour_lines" --overwrite -o'.format(
                    input_contour_lines) + "\n")
            f.write(
                r'v.in.ogr min_area=0.0 snap=-1.0 input={} layer="Basins_4" output={} --overwrite -o'.format(
                    input_watersheds, watersheds_shp) + "\n")

            f.write(
                r'g.region n=1156020.7409 s=1145261.853 e=364958.1745 w=357194.6588 res=30.565022451815462' + "\n")

            f.write(
                r'v.to.rast input=contour_lines layer="contour_lines_with_watershed_id_enerata" type="point,line,area" where="pos_rank in {}" use="val" value=1 memory=300 output={} --overwrite'.format(
                    indizes_protected_areas, selected_contour_lines_ras) + "\n")
            f.write(
                r'v.to.rast input={} layer="Basins_4" type="point,line,area" where="pos_rank in {}" use="val" value=1 memory=300 output={} --overwrite'.format(
                    watersheds_shp, indizes_protected_areas, protected_watersheds_ras) + "\n")
            f.write(r'r.null map={} null=0'.format(selected_contour_lines_ras) + "\n")
            f.write(
                r'r.flow  elevation={} barrier={} flowline={} flowlength={} flowaccumulation={} --overwrite'.format(
                    dem_ras, selected_contour_lines_ras, output_flowline_ras, output_flowlength_ras,
                    output_flowacc_ras) + "\n")
            f.write(r'g.region raster={}'.format(output_flowlength_ras) + "\n")
            f.write(
                r'r.mapcalc --overwrite expression=""{pfactor_ras}" = if({landuse_ras} == 1, (if(!isnull({protected_watersheds_ras}), 0.7 , 0.2)),(if({landuse_ras} == 2, if(!isnull({protected_watersheds_ras}), 0.7 , 0.2), (if({landuse_ras} == 3 || {landuse_ras} == 4 || {landuse_ras} == 5, 1,0)) )) )"'.format(
                    pfactor_ras=p_factor_ras, landuse_ras=landuse_ras,
                    protected_watersheds_ras=protected_watersheds_ras) + "\n")
            f.write(
                r'r.mapcalc --overwrite expression=""{m_ras}" = (((sin({slope_degrees} * {pi_constant}/180) / 0.0896)/ (3 + sin({slope_degrees} * {pi_constant}/180) * 0.8 + 0.56))/(1+ (sin({slope_degrees} * {pi_constant}/180) / 0.0896)/(3 + sin({slope_degrees} * {pi_constant}/180) * 0.8 + 0.56)))"'.format(
                    slope_degrees=slope_degrees, m_ras=m_ras, pi_constant=pi_constant) + "\n")
            f.write(r'r.mapcalc --overwrite expression=""{l}"=({fl}/22.13)^{m}"'.format(
                fl=output_flowlength_ras, m=m_ras, l=lfactor_ras) + "\n")
            f.write(
                r'r.mapcalc --overwrite expression=""{s}"=(if({slope}<5,10 * sin({slope}) + 0.03,(if(5<{slope}<=10,16*sin({slope})-0.55,21.9*sin({slope})-0.96))))"'.format(
                    slope=slope_degrees, s=s_factor_ras) + "\n")
            f.write(
                r'r.mapcalc --overwrite expression=""{ls}"={l}*{s}"'.format(l=lfactor_ras, s=s_factor_ras,
                                                                            ls=lsfactor_ras) + "\n")
            f.write(
                r'r.mapcalc --overwrite expression=""{rusle}"={r}*{k}*{ls}*{c}*{p}"'.format(r=rfactor_ras,
                                                                                            k=kfactor_ras,
                                                                                            ls=lsfactor_ras,
                                                                                            c=cfactor_ras,
                                                                                            p=p_factor_ras,
                                                                                            rusle=rusle_ras) + "\n")

            f.write(r'r.univar -t map={rusle} separator=comma output="{stats}" --overwrite'.format(
                rusle=rusle_ras, stats=csv_stats) + "\n")

            if savemaps is True:
                outputdir = os.path.join(gis_env['GISDBASE'], str(gis_env['LOCATION_NAME']),str(solution._solution_id))
                f.write(
                    r'r.out.gdal -t -m input={} output="{}/flowlength.tif" format=GTiff  createopt=TFW=YES,COMPRESS=LZW --overwrite'.format(
                        output_flowlength_ras, outputdir) + "\n")
                f.write(
                    r'r.out.gdal -t -m input={} output="{}/r_factor.tif" format=GTiff  createopt=TFW=YES,COMPRESS=LZW --overwrite'.format(
                        rfactor_ras, outputdir) + "\n")
                f.write(
                    r'r.out.gdal -t -m input={} output="{}/k_factor.tif" format=GTiff  createopt=TFW=YES,COMPRESS=LZW --overwrite'.format(
                        kfactor_ras, outputdir) + "\n")
                f.write(
                    r'r.out.gdal -t -m input={} output="{}/l_factor.tif" format=GTiff  createopt=TFW=YES,COMPRESS=LZW --overwrite'.format(
                        lfactor_ras, outputdir) + "\n")
                f.write(
                    r'r.out.gdal -t -m input={} output="{}/s_factor.tif" format=GTiff  createopt=TFW=YES,COMPRESS=LZW --overwrite'.format(
                        s_factor_ras, outputdir) + "\n")
                f.write(
                    r'r.out.gdal -t -m input={} output="{}/ls_factor.tif" format=GTiff  createopt=TFW=YES,COMPRESS=LZW --overwrite'.format(
                        lsfactor_ras, outputdir) + "\n")
                f.write(
                    r'r.out.gdal -t -m input={} output="{}/p_factor.tif" format=GTiff  createopt=TFW=YES,COMPRESS=LZW --overwrite'.format(
                        p_factor_ras, outputdir) + "\n")
                f.write(
                    r'r.out.gdal -t -m input={} output="{}/c_factor.tif" format=GTiff  createopt=TFW=YES,COMPRESS=LZW --overwrite'.format(
                        cfactor_ras, outputdir) + "\n")
                f.write(
                    r'r.out.gdal -t -m input={} output="{}/rusle.tif" format=GTiff  createopt=TFW=YES,COMPRESS=LZW --overwrite'.format(
                        rusle_ras, outputdir) + "\n")

                f.write(
                    r'v.extract input=contour_lines  output=selected_contour_lines_shp layer="contour_lines_with_watershed_id_enerata" where="pos_rank in {}"'.format(indizes_protected_areas)+ "\n")

                f.write(
                    r'v.extract input={}  output={} layer="Basins_4" where="pos_rank in {}"'.format(
                        watersheds_shp,selected_watersheds_shp, indizes_protected_areas) + "\n")

                f.write(
                    r'v.out.ogr input={} output="{}/terraces.geojson" format=GeoJSON  --overwrite'.format(
                        selected_contour_lines_shp, outputdir) + "\n")

                f.write(
                    r'v.out.ogr input={} output="{}/protected_watersheds.geojson" format=GeoJSON --overwrite'.format(
                        selected_watersheds_shp, outputdir) + "\n")


        f.close()

        grass7bin_win = r'C:\OSGeo4W64\bin\grass78.bat'
        startcmd = [grass7bin_win, os.path.join(outputdir,"PERMANENT"), '--exec', os.path.join(bash_location,"rusle.sh")]
        print(startcmd)
        p = subprocess.Popen(startcmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        processes.append(p)
        return

    start = time.time()
    processes = []
    outputdirectories = []
    for solution in solutions:
        compute_rusle_per_solution(solution, processes, outputdirectories)

    begintotal = time.time()
    output = [p.wait() for p in processes]
    endtotal = time.time()
    f1 = []
    print(outputdirectories)
    for dir in outputdirectories:
        df = pd.read_csv(os.path.join(dir,'rusle_total.csv'))
        f1.append(df['sum'].iloc[0])


    end = time.time()
    return os.path.join(gis_env['GISDBASE'], str(gis_env['LOCATION_NAME']))