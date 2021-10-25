from pymoo.factory import get_problem
import matplotlib as mpl
import PyQt5

import matplotlib.pyplot as plt
from pymoo.util.plotting import plot

#problem = get_problem("zdt5", normalize=False)
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize
from pymoo.problems.multi.zdt import ZDT5

from pymoo.model.problem import Problem
import autograd.numpy as anp
from pymoo.util.normalization import normalize
import os.path
import sys
import numpy as np
import time
import tempfile
import shutil
import math
import subprocess
import binascii

import pandas as pd
import seaborn as sns

sys.path.append(".")
from .utility_functions import compute_rusle, selected_contour_lines_to_raster, \
    selected_watersheds_to_raster,compute_flow_path_length, compute_ls_factor_map,\
    compute_p_factor_map, compute_rusle, compute_sum_of_raster, raster_to_dataset, setup_grass

gis_env = setup_grass(grass7bin = r'C:\OSGeo4W64\bin\grass78.bat', myepsg = '32637')

def run_optimization():
    class MyProblem(Problem):
    
        # by calling the super() function the problem properties are initialized
        # def __init__(self):
        #     super().__init__(n_var=100,  # nr of variables
        #                      n_obj=2,  # nr of objectives
        #                      n_constr=0,  # nr of constrains
        #                      xl=0.0,  # lower boundaries
        #                      xu=1.0)  # upper boundaries
    
        # the _evaluate function needs to be overwritten from the superclass
        # the method takes two-dimensional NumPy array x with n rows and n columns as input
        # each row represents an individual and each column an optimization variable
    
        def __init__(self, m=1, n=288, normalize=False, **kwargs):
            self.m = m
            self.n = n
            self.normalize = normalize
            super().__init__(n_var=n, **kwargs)
    
        def _calc_pareto_front(self, n_pareto_points=100):
            x = 1 + anp.linspace(0, 1, n_pareto_points) * 30
            pf = anp.column_stack([x, (self.m - 1) / x])
            if self.normalize:
                pf = normalize(pf)
            return pf
    
        def _evaluate(self, X, out, *args, **kwargs):
            generation_id = 0

            def calculate_rusle(solutions,input_dem, input_landuse_raster, input_watersheds, input_contour_lines, input_slope_degrees,input_r_factor, input_k_factor, input_c_factor, savemaps = False, outputdir = None):
                def compute_rusle_per_solution(solution, processes, outputdirectories):
                    random_folder_name = binascii.hexlify(os.urandom(16)).decode('ascii')
                    os.mkdir(os.path.join(gis_env['GISDBASE'], str(gis_env['LOCATION_NAME']),random_folder_name))
                    shutil.copytree(os.path.join(gis_env['GISDBASE'], str(gis_env['LOCATION_NAME']),"PERMANENT"), os.path.join(gis_env['GISDBASE'], str(gis_env['LOCATION_NAME']),random_folder_name,"PERMANENT"))
                    bash_location = os.path.join(gis_env['GISDBASE'], str(gis_env['LOCATION_NAME']),random_folder_name)
                    outputdir = os.path.join(gis_env['GISDBASE'], str(gis_env['LOCATION_NAME']),random_folder_name, "PERMANENT")
                    outputdirectories.append(os.path.join(gis_env['GISDBASE'], str(gis_env['LOCATION_NAME']),random_folder_name))
                    #1. select barriers from selected spatial units (sub-watersheds)
                    indizes_protected_areas = tuple(np.where(solution == True)[0])
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
                    watersheds_shp = "watersheds_shp"
                    m_ras = "m_ras"
                    s_factor_ras = "sfactor_ras"
                    p_factor_ras = "pfactor_ras"
                    lfactor_ras = "lfactor_ras"
                    lsfactor_ras = "lsfactor_ras"
                    rusle_ras = "rusle_ras"
                    csv_stats = os.path.join(gis_env['GISDBASE'], str(gis_env['LOCATION_NAME']),random_folder_name,"rusle_total.csv")
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
                            outputdir = os.path.join(gis_env['GISDBASE'], str(gis_env['LOCATION_NAME']),random_folder_name)
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
                    f.close()

                    grass7bin_win = r'C:\OSGeo4W64\bin\grass78.bat'
                    startcmd = [grass7bin_win, outputdir, '--exec', os.path.join(bash_location,"rusle.sh")]
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
                return f1, end - start, outputdirectories

            def calculate_labour_requirements(solutions, watersheds, labour_requirement_map):
                def compute_labour_requirements_per_solution(solution):
                    tempdir = tempfile.TemporaryDirectory()
                    outputdir = tempdir.name

                    indizes_protected_areas = tuple(np.where(solution == True)[0])
                    extent = '357194.6588,364958.1745,1145261.853,1156020.7409 [EPSG:32637]'
                    active_soil_conservation_raster = selected_watersheds_to_raster(indizes_protected_areas, watersheds,
                                                                                    extent, outputdir)
                    active_soil_conservation_dataset = raster_to_dataset(active_soil_conservation_raster)
                    labour_requirements_dataset = raster_to_dataset(labour_requirement_map)

                    sum_required_labour = np.sum(labour_requirements_dataset[[active_soil_conservation_dataset == 1]])

                    del active_soil_conservation_dataset, labour_requirements_dataset
                    return sum_required_labour

                start = time.time()
                f2 = [compute_labour_requirements_per_solution(solution) for solution in solutions]
                end = time.time()
                return f2, end - start
    
            solutions = [X[:, :self.n]][0]

            input_dem = r'C:/Users/morit/AppData/Local/Temp/grassdata/mytemploc_utm32n/DEM_Enerata_filled.tif'
            input_contour_lines = r'C:/Users/morit/AppData/Local/Temp/grassdata/mytemploc_utm32n/contour_lines_with_watershed_id_enerata.shp'
            input_watersheds = r'C:/Users/morit/AppData/Local/Temp/grassdata/mytemploc_utm32n/Basins_4.shp'
            input_landuse_raster = r"C:/Users/morit/AppData/Local/Temp/grassdata/mytemploc_utm32n/landuse_raster.tif"
            input_r_factor = r"C:/Users/morit/AppData/Local/Temp/grassdata/mytemploc_utm32n/r_factor_clipped.tif"
            input_k_factor = r"C:/Users/morit/AppData/Local/Temp/grassdata/mytemploc_utm32n/k_factor_clipped.tif"
            input_c_factor = r"C:/Users/morit/AppData/Local/Temp/grassdata/mytemploc_utm32n/c_factor_clipped.tif"
            input_slope_degrees = r"C:/Users/morit/AppData/Local/Temp/grassdata/mytemploc_utm32n/Slope_degrees.tif"
            labour_requirements = r"C:\Users\morit\OneDrive - Universität Münster\PhD\Kooperation_GIZ\Data\Optimization_Enerata\Output Enerata\labour_requirement_data\labour_requirements.tif"

            f1, timef1, outputdirectories = calculate_rusle(solutions, input_dem, input_landuse_raster, input_watersheds, input_contour_lines, input_slope_degrees,input_r_factor,input_k_factor,input_c_factor)
            f2, timef2 = calculate_labour_requirements(solutions, input_watersheds, labour_requirements)
            for dir in outputdirectories:
                shutil.rmtree(dir, ignore_errors=False, onerror=None)
            print("Generation: "+ str(generation_id))
            generation_id += 1
            print("Execution time objective function 1: "+ str(timef1))
            print("Execution time objective function 1: "+ str(timef2))
            if self.normalize:
                f1 = normalize(f1, 1, self.n + 1)
                f2 = normalize(f2, 1, self.n + 1)
            out["F"] = anp.column_stack([f1, f2])

    problem = MyProblem(parallelization=True)

    algorithm = NSGA2(
        pop_size=4,
        sampling=get_sampling("bin_random"),
        crossover=get_crossover("bin_hux"),
        mutation=get_mutation("bin_bitflip"),
        eliminate_duplicates=True)
    
    res = minimize(problem,
                    algorithm,
                    ('n_gen', 2),
                    verbose=True,
                    pf=problem.pareto_front(use_cache=False),
                    save_history=True)

    """
    res.X design space values are
    res.F objective spaces values
    res.G constraint values
    res.CV aggregated constraint violation
    res.algorithm algorithm object
    res.pop final population object
    res.history history of algorithm object. (only if save_history has been enabled during the algorithm initialization)
    res.time the time required to run the algorithm
    """

    import matplotlib.pyplot as plt
    # plt.scatter(res.F[:,0], res.F[:,1])
    # plt.show()

    F = []

    # iterate over the deepcopies of algorithms
    for algorithm in res.history:
        # retrieve the optimum from the algorithm
        opt = algorithm.opt
        _F = opt.get("F")
        print("F: ")
        print(_F)
        F.append(_F)

    # make an array of the number of generations
    n_gen = np.array(range(1, len(F) + 1))

    print(n_gen)
    print(F)
    generations = []
    for i in range(len(F)):
        for j in range(F[i].shape[0]):
            generations.append([i, F[i][j][0],F[i][j][1]])
    convergence_df = pd.DataFrame(data=generations, columns =["generation_id", "RUSLE", "Labour requirements"])
    final_population_df = pd.DataFrame(data=res.pop)

    convergence_df.to_pickle(os.path.join(gis_env['GISDBASE'], str(gis_env['LOCATION_NAME']), "convergence_df.pkl"))
    final_population_df.to_pickle(os.path.join(gis_env['GISDBASE'], str(gis_env['LOCATION_NAME']), "final_population_df.pkl"))
    sns.scatterplot(data=convergence_df, x="RUSLE", y= "Labour requirements", hue="generation_id")
    plt.show()
    print(convergence_df)
