import itertools
import pickle
import os
import shutil
import numpy as np
try:
    from .optimization_under_uncertainty import calculate_labour_requirements_under_uncertainty, calculate_rusle_under_uncertainty
    from .utility_functions import setup_grass,compute_extent,compute_crs,load_raster_dataset,get_area_in_ha
except:
    pass
import os
import subprocess
import shutil
import sys
import math
import time
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy import special as sp
import math
#import geostatspy
import pickle



import pymoo
from pymoo.factory import get_problem, get_reference_directions, get_visualization
from pymoo.util.plotting import plot
from pymoo.util.nds import fast_non_dominated_sort
from pymoo.util.dominator import Dominator
from pymoo.util.function_loader import load_function

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem, get_sampling, get_crossover, get_mutation, get_selection
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

from pymoo.factory import get_problem
import matplotlib as mpl
import matplotlib.pyplot as plt
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
import pathlib
import scipy.stats as st
from scipy import special as sp
import math
import pickle

from pymoo.util.plotting import plot
#from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_crossover, get_mutation, get_sampling, get_selection
from pymoo.optimize import minimize
from pymoo.problems.multi.zdt import ZDT5
from pymoo.model.problem import Problem
import autograd.numpy as anp
from pymoo.util.normalization import normalize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
#from .nsga2Unc import NSGA2Unc
class Solution:
    _id = 0

    def __init__(self, representation, objective_values):
        self._solution_id = Solution._id
        Solution._id += 1
        self.representation = representation
        self.objective_values = objective_values


def compute_optimal_number_of_realizations_in_optimization():
    gis_env = setup_grass(grass7bin=r'C:\OSGeo4W64\bin\grass78.bat', myepsg='32637')

    max_n_realizations =47

    #missing steps:
    #Lösung auswählen
    #Watersheds vorbereiten

    extreme_solution_max_protection = np.array([True for i in range(146)])
    extreme_solution_min_protection = np.array([False for i in range(146)])
    intermediate_solution = []
    for i in range(146):
        if i%2 == 0:
            intermediate_solution.append(True)
        else:
            intermediate_solution.append(False)
    intermediate_solution = np.array(intermediate_solution)
    solutions = [extreme_solution_max_protection,extreme_solution_min_protection,intermediate_solution]
    wrkdir = r"C:/Users/morit/AppData/Local/Temp/grassdata/gumobila"
    input_dem = wrkdir+"//"+'DEM_Gumobila2.tif'
    extent = compute_extent(input_dem)
    crs = compute_crs(input_dem)
    input_contour_lines_raster = wrkdir+"//"+"contourlines_raster_with_id.tif"
    input_landuse_raster = wrkdir+"//"+"landuse_raster.tif"
    input_r_factors = wrkdir+"//"+"r_factor_realizations"
    input_k_factors = wrkdir+"//"+"k_factor_realizations"
    input_c_factor = wrkdir+"//"+"c_factor_clipped.tif"
    input_slope_degrees = wrkdir+"//"+"Slope_degrees.tif"
    labour_requirements_dir = wrkdir+"//"+"labor_requirements"
    #load only dataset of watersheds raster to accelerate computation
    input_watersheds_raster = wrkdir+"//"+"watersheds_raster_with_id.tif"
    # load dataset of watersheds raster to accelerate computation
    input_watersheds_dataset = load_raster_dataset(input_watersheds_raster)
    input_watersheds_shp = wrkdir+"//"+ "watersheds_with_id.shp"
    input_contour_lines_shp = wrkdir+"//"+"contour_lines_with_watershed_id.shp"

    # load dataset of labor requirement raster to accelerate computation
    input_labour_requirements = []
    for filename in os.listdir(labour_requirements_dir):
        if filename.endswith(".tif"):
            input_labour_requirements.append(load_raster_dataset(os.path.join(labour_requirements_dir, filename)))
    f1 = []
    f2 = []
    interval = 1
    study_areasize_ha = get_area_in_ha(input_watersheds_shp)
    print("Study area size in ha: "+ str(study_areasize_ha))
    execution_times = []
    for i in range(0,47, interval):
        selected_realization_ids = [j for j in range(i)]
        obj1, timef1, outputdirectories = calculate_rusle_under_uncertainty(solutions, input_dem,
                                                                          input_landuse_raster, input_watersheds_raster,
                                                                          input_contour_lines_raster, input_slope_degrees,
                                                                          input_r_factors,
                                                                          input_k_factors, input_c_factor,
                                                                          selected_realization_ids,input_watersheds_shp, input_contour_lines_shp,extent,crs,study_areasize_ha,
                                                                            savemaps=False,
                                                                          )
        [f1.append(obj1[i]) for i in range(len(obj1))]
        obj2, timef2 = calculate_labour_requirements_under_uncertainty(solutions, input_watersheds_dataset,
                                                                     input_labour_requirements,selected_realization_ids,study_areasize_ha)
        [f2.append(obj2[i]) for i in range(len(obj2))]
        print("Execution time objective function 1: " + str(timef1))
        print("Execution time objective function 2: " + str(timef2))
        execution_times.append(timef1+timef2)
        #for dir in outputdirectories:
            #shutil.rmtree(dir, ignore_errors=False, onerror=None)

    population_with_reshaped_objectivevalues = []
    for i in range(len(f1)):
        population_with_reshaped_objectivevalues.append([f1[i], f2[i]])


    outputdir = r"C:\Users\morit\OneDrive - Universität Münster\PhD\Kooperation_GIZ\Data\Optimization_Gumobila\optimization_result"
    with open(os.path.join(outputdir, 'optimal_number_realizations.pkl'), 'wb') as handle:
        pickle.dump(population_with_reshaped_objectivevalues, handle, protocol=pickle.HIGHEST_PROTOCOL)


    with open(os.path.join(outputdir, 'optimal_number_realizations_execution_times.pkl'), 'wb') as handle:
        pickle.dump(execution_times, handle, protocol=pickle.HIGHEST_PROTOCOL)


outputdir = r"C:\Users\morit\OneDrive - Universität Münster\PhD\Kooperation_GIZ\Data\Optimization_Gumobila\optimization_result"
with open(os.path.join(outputdir, 'optimal_number_realizations_execution_times.pkl'), 'rb') as handle:
    times = pickle.load(handle)
with open(os.path.join(outputdir, 'optimal_number_realizations.pkl'), 'rb') as handle:
    solutions = pickle.load(handle)
plt.scatter([i for i in range(47)], times)
plt.show()
print()

obj1_values = []
obj2_values = []
solution_ids = []
realization_ids = []

objvalues_max_protection = []
objvalues_min_protection = []
objvalues_intermediate_protection = []
nr_realizations = []
for i in range(0,len(solutions),3):
    for realization_id in range(len(solutions[i][0])):
        objvalues_max_protection.append([solutions[i][0][realization_id],solutions[i][1][realization_id]])
        objvalues_min_protection.append([solutions[i+1][0][realization_id], solutions[i+1][1][realization_id]])
        objvalues_intermediate_protection.append([solutions[i + 2][0][realization_id], solutions[i + 2][1][realization_id]])
        nr_realizations.append(len(solutions[i][0]))

df_max_protection = pd.DataFrame(
    {'obj1': np.array([i[0] for i in objvalues_max_protection]), 'obj2': np.array([i[1] for i in objvalues_max_protection]), 'nr_realizations': np.array(nr_realizations)})
#filter out invalid solutions
df_max_protection = df_max_protection[df_max_protection['obj1']<100]

df_min_protection = pd.DataFrame(
    {'obj1': np.array([i[0] for i in objvalues_min_protection]), 'obj2': np.array([i[1] for i in objvalues_min_protection]), 'nr_realizations': np.array(nr_realizations)})
#filter out invalid solutions
df_min_protection = df_min_protection[df_min_protection['obj1']<100]

df_intermediate_protection = pd.DataFrame(
    {'obj1': np.array([i[0] for i in objvalues_intermediate_protection]), 'obj2': np.array([i[1] for i in objvalues_intermediate_protection]), 'nr_realizations': np.array(nr_realizations)})
#filter out invalid solutions
df_intermediate_protection = df_intermediate_protection[df_intermediate_protection['obj1']<100]


#df_min_protection = pd.DataFrame(
#    {'obj1': np.array(obj1_values), 'obj2': np.array(obj2_values), 'sol_id': np.array(solution_ids)})
means_max_protection = df_max_protection.groupby(['nr_realizations']).mean()
std_max_protection = df_max_protection.groupby(['nr_realizations']).std()

means_min_protection = df_min_protection.groupby(['nr_realizations']).mean()
std_min_protection = df_min_protection.groupby(['nr_realizations']).std()

means_intermediate_protection = df_intermediate_protection.groupby(['nr_realizations']).mean()
std_intermediate_protection = df_intermediate_protection.groupby(['nr_realizations']).std()

fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=[12,4])
# selection_size = 40
# ax0.errorbar(np.array([i for i in range(selection_size)]), means_max_protection["obj1"][:selection_size], yerr=std_max_protection["obj1"][:selection_size], fmt='-o',markersize=2. )
# #ax0.errorbar(np.array([i for i in range(selection_size)]), means_intermediate_protection["obj1"][:selection_size], yerr=std_intermediate_protection["obj1"][:selection_size], fmt='-o',markersize=2.)
# #ax0.errorbar(np.array([i for i in range(selection_size)]), means_min_protection["obj1"][:selection_size], yerr=std_min_protection["obj1"][:selection_size], fmt='-o',markersize=2.)
# ax1.errorbar(np.array([i for i in range(selection_size)]), means_max_protection["obj2"][:selection_size], yerr=std_max_protection["obj2"][:selection_size], fmt='-o',markersize=2. )

dataobj1 = []
dataobj2 = []
for i in range(5,36):
    dataobj1.append(df_max_protection[df_max_protection["nr_realizations"]<i]["obj1"])
    dataobj2.append(df_max_protection[df_max_protection["nr_realizations"]< i]["obj2"])
ax0.boxplot(dataobj1)
ax1.boxplot(dataobj2)
plt.xticks(list(range(1,33)),list(range(5,36)))
ax0.set_title('Max protection')
ax0.set_ylabel('Soil loss in t/ha/year')
ax1.set_ylabel('Labour requirement in labor days/year')
plt.xlabel("Nr. of realizations")
plt.legend([],[], frameon=False)
plt.savefig(r"C:\Users\morit\OneDrive - Universität Münster\PhD\Kooperation_GIZ\Data\Optimization_Gumobila\optimization_result\optimal_nr_realizations.svg", format="svg")
plt.show()

fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=[12,4])
dataobj1 = []
dataobj2 = []
for i in range(5,30):
    dataobj1.append(df_min_protection[df_min_protection["nr_realizations"]<i]["obj1"])
    dataobj2.append(df_min_protection[df_min_protection["nr_realizations"] < i]["obj2"])
ax0.boxplot(dataobj1)
ax1.boxplot(dataobj2)
ax0.set_title('Min protection')
ax0.set_ylabel('Soil loss in t/ha/year')
ax1.set_ylabel('Labour requirement in labor days/year')
plt.xlabel("Nr. of realizations")
xi = list(range(5,30))
plt.xticks(list(range(25)),list(range(5,30)))
plt.legend([],[], frameon=False)
plt.show()
