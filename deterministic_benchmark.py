import itertools
import pickle
import os
import shutil
import numpy as np
from .optimization_under_uncertainty import calculate_labour_requirements_under_uncertainty, calculate_rusle_under_uncertainty
from .utility_functions import setup_grass,compute_extent,compute_crs,load_raster_dataset,get_area_in_ha
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

n_survive = 25
#compute stats of solutions with uncertain objective values: mean, std, 5% and 95% intervals
def compute_stats_of_uncertain_objective_values(uncertain_pf):
    means = []
    intervals = []
    stds = []
    #compute confidence intervals of solutions
    for solution_id in range(len(uncertain_pf)):
        solution = np.array(uncertain_pf[solution_id])
        #obj1
        mean_obj_1,mean_obj_2 = np.mean(solution[0]),np.mean(solution[1])
        std_obj_1, std_obj_2 = np.std(solution[0]),np.std(solution[1])
        interval_obj_1 = st.t.interval(0.95, solution[0].shape[0]-1, loc=mean_obj_1, scale=st.sem(solution[0]))
        interval_obj_2 = st.t.interval(0.95, solution[1].shape[0]-1, loc=mean_obj_2, scale=st.sem(solution[1]))
        means.append([mean_obj_1, mean_obj_2])
        stds.append([std_obj_1,std_obj_2])
        intervals.append([interval_obj_1,interval_obj_2])
    return means, stds, intervals

#functions for computing the probabilistic dominance by Eskandari and Geiger 2009
def compute_probabilistic_dominance(mean_solution_A, std_solution_A,mean_solution_B, std_solution_B):
    def qfunc(x):
        # Q-function, approximation of the Q-function suggested by Borjesson and Sundberg (1979)
        # reference: https://en.wikipedia.org/wiki/Q-function, scipy error function, https://stackoverflow.com/questions/56075838/how-to-generate-the-values-for-the-q-function
        return 0.5 - 0.5 * sp.erf(x / math.sqrt(2))
    # H. Eskandari, C.D. Geiger 2009 (DOI 10.1007/s10732-008-9077-z), p. 566
    # P = 1 − Q(μA − μB/ sqrt(σA^2+σB^2)
    return 1 - qfunc((mean_solution_A - mean_solution_B) / math.sqrt(np.power(std_solution_A, 2) + np.power(std_solution_B, 2)))

#computes the dominance relationship between solutuins following the methodology of H. Eskandari, C.D. Geiger 2009 (DOI 10.1007/s10732-008-9077-z), p. 564 ff.
#3 possibilities: No dominance, significant dominance, probabilistic dominance
def compute_probability_of_domination(intervals_sol1, intervals_sol2, means_sol1, means_sol2, stds_sol1, stds_sol2):
    # inputs:two-dimensional lists.
    # First level (all inputs): Objectives, Objective 1 = [0] or Objective 2 = [1].
    # Second level of interval inputs: Lower bound[0], Upper bound [1]

    # compute_level_of_dominance
    # no dominance: 0., significant dominance: 1., Probabilistic Dominance P otherwise
    level_of_dominance = None

    # check for no dominance
    # "Solution A does not dominate solution B when at least one lower bound of the solution A confidence
    #  interval is larger than the corresponding upper bound of solution B"
    if intervals_sol1[0][0] > intervals_sol2[0][1] or intervals_sol1[1][0] > intervals_sol2[1][1]:
        level_of_dominance = 0.

    # check for significant dominance
    # "Second, solution A significantly dominates solution B when all upper bounds of the solution A confidence
    #  interval are less than the corresponding upper bound of solution B"
    elif intervals_sol1[0][1] < intervals_sol2[0][1] and intervals_sol1[1][1] < intervals_sol2[1][1]:
        level_of_dominance = 1.

    # check for probabilistic dominance:
    # "In the third case, solution A probabilistically dominates solution B with a certain probability
    #  when all lower bounds of  the solution A confidence intervals are less than the corresponding upper bounds of
    #  solution B"
    elif intervals_sol1[0][0] < intervals_sol2[0][0] and intervals_sol1[1][0] < intervals_sol2[1][0]:
        # obj1
        prob_dominance_sol_1_obj1 = compute_probabilistic_dominance(means_sol1[0], stds_sol1[0],means_sol2[0], stds_sol2[0])
        #prob_dominance_sol_2_obj1 = 1 - prob_dominance_sol_1_obj1
        # obj2
        prob_dominance_sol_1_obj2 = 1 - compute_probabilistic_dominance(means_sol1[1], stds_sol1[1],means_sol2[1], stds_sol2[1])
        #prob_dominance_sol_2_obj2 = 1 - prob_dominance_sol_1_obj2

        level_of_dominance = prob_dominance_sol_1_obj1 * prob_dominance_sol_1_obj2

    # check for stochastic dominance
    # "Solution A stochastically dominates (is better than) solution B if mean f(A) is less than mean f(B) for each objective function
    elif means_sol1[0] < means_sol2[0] and means_sol1[1] < means_sol2[1]:
        level_of_dominance = 1

    else:
        # obj1
        prob_dominance_sol_1_obj1 = compute_probabilistic_dominance(means_sol1[0], stds_sol1[0], means_sol2[0],
                                                                    stds_sol2[0])
        # prob_dominance_sol_2_obj1 = 1 - prob_dominance_sol_1_obj1
        # obj2
        prob_dominance_sol_1_obj2 = 1 - compute_probabilistic_dominance(means_sol1[1], stds_sol1[1], means_sol2[1],
                                                                        stds_sol2[1])
        # prob_dominance_sol_2_obj2 = 1 - prob_dominance_sol_1_obj2

        level_of_dominance = prob_dominance_sol_1_obj1 * prob_dominance_sol_1_obj2

    return level_of_dominance

#here, the matrix of all solutions is computed with their dominance relationship derived from the function above. The other method (not Eskandari Geiger) is not fully implemented yet
def compute_probabilistic_dominance_matrix(uncertain_pf,means, stds, intervals, method = "Eskandari_Geiger"):
    nr_solutions = len(uncertain_pf)
    probabilistic_dominance_matrix = np.empty([nr_solutions, nr_solutions])
    for i in range(nr_solutions):
        for j in range(nr_solutions):
            # compute probability of solution A dominating solution B and add to matrix
            if i == j:
                probabilistic_dominance_matrix[i, j] = None
            else:
                if method == "Eskandari_Geiger":
                    probabilistic_dominance_matrix[i, j] = compute_probability_of_domination(intervals[i],intervals[j],means[i], means[j], stds[i], stds[j])
                else:
                    probabilistic_dominance_matrix[i, j] = compute_probabilistic_dominance_value(means[i], stds[i], means[j], stds[j])
    return probabilistic_dominance_matrix

# here, the functions above are called and the ranking and fitness assessment is executed. The optimization it is used for is similar to NSGA2 and called SPGA
def ranking_and_fitness_assignment_SPGA(uncertain_pf, n_survive):
    # ranking and fitness evaluation by H. Eskandari, C.D. Geiger 2009 (DOI 10.1007/s10732-008-9077-z), p. 569 f.

    # Preparation: compute mean, standarddeviations and 5%,95% intervals
    means, stds, intervals = compute_stats_of_uncertain_objective_values(uncertain_pf)
    #Step 1: identify all stochastically non-dominated solutions (rank 1)
    #Stochastic dominance is  evaluated by using the sample average
    nds = NonDominatedSorting()
    ranked_on_sample_means = nds.do(np.array(means), return_rank=True)
    stochastically_non_dominated= ranked_on_sample_means[0][0]

    #Step 2:
    # "All stochastically dominated,
    # which comprise the second rank, are assigned an expected strength value E[S]
    # indicating the summation of the probabilities that it dominates other solutions"

    probabilistic_dominance_matrix = compute_probabilistic_dominance_matrix(uncertain_pf,means, stds, intervals, method = "Eskandari_Geiger")
    # cumulative dominance probability: sum per row of matrix, called E[S]
    cumulative_dominance_Es = np.nansum(probabilistic_dominance_matrix, axis=1)

    #Step 3: compute expected fitness calles E[f]:
    # "a fitness value is assigned to each dominated solution x is equal to the
    # summation of the expected strength values (E[S]) of all solutions it stochastically dominates
    # minus the summation of the expected strength values of all solutions by which it
    # is stochastically dominated."
    expected_fitnesses = []
    for solution_i_id in range(len(uncertain_pf)):
        sum_Es_solution_i_stochastically_dominates = 0
        sum_Es_solution_i_is_stochastically_dominated_by = 0
        if solution_i_id not in (ranked_on_sample_means[0][0]):
            #get row of solution id and get solutions with higher stochastic dominance rank
            dominated_by_i = probabilistic_dominance_matrix[solution_i_id][ranked_on_sample_means[1][solution_i_id]<ranked_on_sample_means[1]][:]
            sum_dominated_by_i = np.nansum(dominated_by_i)
            # get column of solution id and get solutions with lower stochastic dominance rank
            dominate_i = probabilistic_dominance_matrix[ranked_on_sample_means[1][solution_i_id]>ranked_on_sample_means[1]][:,solution_i_id]
            sum_dominate_i = np.nansum(dominate_i)
            expected_fitnesses.append([solution_i_id, sum_dominated_by_i-sum_dominate_i])

    #derive selected solutions of rank 2 with highest fitness values
    # total population size: 20
    #get number of popsize -  amount non-dominated (rank 1)
    open_spots = n_survive - len(stochastically_non_dominated)
    # identify solutions that have highest fitness for open spots
    if open_spots > 0:
        expected_fitnesses_numpy = np.array(expected_fitnesses)
        sorted_expected_fitnesses_of_dominated_solutions = expected_fitnesses_numpy[expected_fitnesses_numpy[:, 1].argsort()][::-1]
        sorted_expected_fitnesses_of_dominated_solutions[:,0] = sorted_expected_fitnesses_of_dominated_solutions[:,0].astype(int)
    else:
        sorted_expected_fitnesses_of_dominated_solutions = []
    return stochastically_non_dominated, sorted_expected_fitnesses_of_dominated_solutions


def execute_deterministic_benchmark():
    gis_env = setup_grass(grass7bin=r'C:\OSGeo4W64\bin\grass78.bat', myepsg='32637')

    n=10
    possible_gene_combinations = list(map(np.array, itertools.product([False, True], repeat=n)))


    wrkdir = r"C:/Users/morit/AppData/Local/Temp/grassdata/mytemploc_utm32n2"
    input_dem = wrkdir+"//"+'DEM_Enerata_filled.tif'
    extent = compute_extent(input_dem)
    crs = compute_crs(input_dem)
    input_contour_lines_raster = wrkdir+"//"+"contour_lines_with_watershed_id.tif"
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
    input_watersheds_shp = wrkdir+"//"+ "Basins_4.shp"
    input_contour_lines_shp = wrkdir+"//"+"contour_lines_with_watershed_id_enerata.shp"
    selected_realization_ids = [0, 1, 2, 3, 4]
    # load dataset of labor requirement raster to accelerate computation
    input_labour_requirements = []
    for filename in os.listdir(labour_requirements_dir):
        if filename.endswith(".tif"):
            input_labour_requirements.append(load_raster_dataset(os.path.join(labour_requirements_dir, filename)))
    f1 = []
    f2 = []
    interval = 8
    study_areasize_ha = get_area_in_ha(input_watersheds_shp)
    print("Study area size in ha: "+ str(study_areasize_ha))
    for i in range(0,len(possible_gene_combinations), interval):
        solutions = possible_gene_combinations[i:i+interval]
        obj1, timef1, outputdirectories = calculate_rusle_under_uncertainty(solutions, input_dem,
                                                                          input_landuse_raster, input_watersheds_raster,
                                                                          input_contour_lines_raster, input_slope_degrees,
                                                                          input_r_factors,
                                                                          input_k_factors, input_c_factor,
                                                                          selected_realization_ids,input_watersheds_shp, input_contour_lines_shp,extent,crs,study_areasize_ha,
                                                                            savemaps=True,
                                                                          )
        [f1.append(obj1[i]) for i in range(len(obj1))]
        obj2, timef2 = calculate_labour_requirements_under_uncertainty(solutions, input_watersheds_dataset,
                                                                     input_labour_requirements,selected_realization_ids,study_areasize_ha)
        [f2.append(obj2[i]) for i in range(len(obj2))]
        print("Execution time objective function 1: " + str(timef1))
        print("Execution time objective function 2: " + str(timef2))
        #for dir in outputdirectories:
            #shutil.rmtree(dir, ignore_errors=False, onerror=None)

    population_with_reshaped_objectivevalues = []
    for i in range(len(f1)):
        population_with_reshaped_objectivevalues.append([f1[i], f2[i]])


    outputdir = r"C:\Users\morit\OneDrive - Universität Münster\PhD\Kooperation_GIZ\Data\Optimization_Enerata\Optimization_result"
    with open(os.path.join(outputdir, 'deterministic_solutions.pkl'), 'wb') as handle:
        pickle.dump(population_with_reshaped_objectivevalues, handle, protocol=pickle.HIGHEST_PROTOCOL)

#visualization





    ####    visualization       #####

benchmark_dir = r"C:\Users\morit\OneDrive - Universität Münster\PhD\Kooperation_GIZ\Data\Optimization_Enerata\Optimization_result"
with open(os.path.join(benchmark_dir, 'deterministic_solutions.pkl'), 'rb') as handle:
    benchmark = pickle.load(handle)

# means, stds, intervals = compute_stats_of_uncertain_objective_values(pop_uncertain_obj_values)
#         # Step 1: identify all stochastically non-dominated solutions (rank 1)
#         # Stochastic dominance is  evaluated by using the sample average
#         nds = NonDominatedSorting()
#         ranked_on_sample_means = nds.do(np.array(means), return_rank=True)
#         stochastically_non_dominated = ranked_on_sample_means[0][0]

survivors = []
stochastically_non_dominated_solutions, sorted_expected_fitnesses_of_dominated_solutions = ranking_and_fitness_assignment_SPGA(
    benchmark, n_survive)
[survivors.append(i) for i in stochastically_non_dominated_solutions]

for i in range(n_survive - len(survivors)):
    survivors.append(int(sorted_expected_fitnesses_of_dominated_solutions[i][0]))
obj1_values = []
obj2_values = []
solution_ids = []
for i in range(len(survivors)):
    for realization_id in range(len(benchmark[0][0])):
        obj1_values.append(benchmark[survivors[i]][0][realization_id])
        obj2_values.append(benchmark[survivors[i]][1][realization_id])
        solution_ids.append(i)
scattered_points = pd.DataFrame(
    {'obj1': np.array(obj1_values), 'obj2': np.array(obj2_values), 'sol_id': np.array(solution_ids)})
sns.scatterplot(data=scattered_points, x='obj1', y='obj2',label='Deterministic solutions',s=70)

outputdir = r"C:\Users\morit\OneDrive - Universität Münster\PhD\Kooperation_GIZ\Data\Optimization_Enerata\Output Enerata\pareto_fronts\test"

#was run with popsize 30, generations = 100
with open(os.path.join(outputdir, 'all_populations8.pkl'), 'rb') as handle:
    populations = pickle.load(handle)

#was run with popsize 10, generations = 40
#with open(os.path.join(outputdir, 'all_populations9.pkl'), 'rb') as handle:
    #populations = pickle.load(handle)


i=0
obj2_min = 0
obj2_max = 60
obj1_min = 0
obj1_max = 60

pop_id = 0
selected_pops=[99]
for pop in populations:
    if pop_id in selected_pops:
        #final_population = populations[-1]
        pop_objective_values = [F for F in pop[0]]

        # correction to soil loss per ha, total size of watershed polygons is 6872.8406 ha
        pop_objective_values = [[F[0], F[1]] for F in
                                             pop_objective_values]
        population_genes = [X for X in pop[1]]
        # final_population_metadata = [elem.data for elem in final_population_df[0]]
        optimal_solutions = []

        for i in range(len(pop_objective_values)):
            optimal_solutions.append(Solution(population_genes[i], pop_objective_values[i]))
        i = 0
        obj1_values = []
        obj2_values = []
        solution_ids = []

        for solution in optimal_solutions:
            for realization_id in range(len(solution.objective_values[0])):
                obj1_values.append(solution.objective_values[0][realization_id])
                obj2_values.append(solution.objective_values[1][realization_id])
                solution_ids.append(i)
            i += 1
        scattered_points = pd.DataFrame(
            {'obj1': np.array(obj1_values), 'obj2': np.array(obj2_values), 'sol_id': np.array(solution_ids)})
        sns.scatterplot(data=scattered_points, x='obj1', y='obj2',s=20, label = 'Optimization, generation {}'.format(str(pop_id)))
    pop_id +=1
plt.xlabel("Soil loss in t/ha/year")
plt.ylabel("Labour days/ha")
plt.show()
