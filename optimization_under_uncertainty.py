from pymoo.factory import get_problem
import matplotlib as mpl
import PyQt5
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
from .nsga2Unc import NSGA2Unc

sys.path.append(".")
from .utility_functions import compute_rusle, selected_contour_lines_to_raster,compute_extent, compute_crs, \
    selected_watersheds_to_raster, compute_flow_path_length, compute_ls_factor_map, \
    compute_p_factor_map, compute_rusle, compute_sum_of_raster, raster_to_dataset, setup_grass,load_raster_dataset, \
    get_area_in_ha, get_nr_watersheds

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
    if n_survive > uncertain_pf.shape[0]:
        n_survive = uncertain_pf.shape[0]
    open_spots = n_survive - len(stochastically_non_dominated)
    # identify solutions that have highest fitness for open spots
    if open_spots > 0 and len(expected_fitnesses) > 0:
        expected_fitnesses_numpy = np.array(expected_fitnesses)
        sorted_expected_fitnesses_of_dominated_solutions = expected_fitnesses_numpy[expected_fitnesses_numpy[:, 1].argsort()][::-1]
        sorted_expected_fitnesses_of_dominated_solutions[:,0] = sorted_expected_fitnesses_of_dominated_solutions[:,0].astype(int)
    else:
        sorted_expected_fitnesses_of_dominated_solutions = []
    return stochastically_non_dominated, sorted_expected_fitnesses_of_dominated_solutions

#function for plotting the Pareto front
def plot_pareto_front(pf_uncertain, stochastically_non_dominated_solutions, sorted_expected_fitnesses_of_dominated_solutions, population_size,  generation_id=None,axis_ranges=None, outputdir=None):
  open_spots = population_size - len(stochastically_non_dominated_solutions)
  drawing_list = []
  for i in range(open_spots,len(sorted_expected_fitnesses_of_dominated_solutions)):
      for j in range(len(pf_uncertain[0][0])):
          # rank, id, objective values
          if len(sorted_expected_fitnesses_of_dominated_solutions) > 0:
            drawing_list.append(["Dominated (3)",int(sorted_expected_fitnesses_of_dominated_solutions[i][0]), pf_uncertain[int(sorted_expected_fitnesses_of_dominated_solutions[i][0])][0][j], pf_uncertain[int(sorted_expected_fitnesses_of_dominated_solutions[i][0])][1][j]])
  for i in range(open_spots):
      for j in range(len(pf_uncertain[0][0])):
          #rank, id, objective values
          if len(sorted_expected_fitnesses_of_dominated_solutions)>0:
            drawing_list.append(["Probabilistically dominating (2)",int(sorted_expected_fitnesses_of_dominated_solutions[i][0]), pf_uncertain[int(sorted_expected_fitnesses_of_dominated_solutions[i][0])][0][j], pf_uncertain[int(sorted_expected_fitnesses_of_dominated_solutions[i][0])][1][j]])
  for i in range(len(stochastically_non_dominated_solutions)):
      for j in range(len(pf_uncertain[0][0])):
          # rank, id, objective values
          drawing_list.append(["Stochastically dominating (1)",i, pf_uncertain[stochastically_non_dominated_solutions[i]][0][j], pf_uncertain[stochastically_non_dominated_solutions[i]][1][j]])
  df = pd.DataFrame(drawing_list, columns=["rank","sol_id", "obj1", "obj2"])

  plt.scatter(x=list(df[df['rank'] == "Stochastically dominating (1)"]["obj1"]),
              y=list(df[df['rank'] == "Stochastically dominating (1)"]["obj2"]),
              marker='X', c=list(df[df['rank'] == "Stochastically dominating (1)"]["sol_id"]),
              cmap='tab10', label="Stochastically dominating (1)")

  plt.scatter(x=list(df[df['rank'] == "Probabilistically dominating (2)"]["obj1"]),
              y = list(df[df['rank'] == "Probabilistically dominating (2)"]["obj2"]),
              marker='+',c=list(df[df['rank'] == "Probabilistically dominating (2)"]["sol_id"]),
              cmap= 'tab20c', label = "Probabilistically dominating (2)")

  plt.scatter(x=list(df[df['rank'] == "Dominated (3)"]["obj1"]),
              y=list(df[df['rank'] == "Dominated (3)"]["obj2"]),
              marker='.', c=list(df[df['rank'] == "Dominated (3)"]["sol_id"]),
              cmap='Pastel2', label="Dominated (3)")
  plt.rcParams['figure.figsize'] = [16, 8]
  plt.rcParams['figure.dpi'] = 200 # 200 e.g. is really fine, but slower
  if axis_ranges is not None:
      plt.xlim(axis_ranges[0], axis_ranges[1])
      plt.ylim(axis_ranges[2], axis_ranges[3])
  plt.legend("Pareto front Generation "+ str(generation_id))
  plt.title('Minimum Message Length')
  #plt.show()
  plt.savefig(os.path.join(outputdir,'ParetoFront_Gen{}.png'.format(generation_id)))
  plt.clf()

gis_env = setup_grass(grass7bin=r'C:\OSGeo4W64\bin\grass78.bat', myepsg='32637')


def calculate_rusle_under_uncertainty(solutions, input_dem, input_landuse_raster, input_watersheds,
                                      input_contour_lines,
                                      input_slope_degrees, input_r_factor_dir, input_k_factor_dir,
                                      input_c_factor, selected_realization_ids,watersheds_shp, contourlines_shp,extent,crs,study_area_ha,
                                      savemaps=False,
                                      outputdir=None):
    def compute_rusle_per_solution_under_uncertainty(solution, processes, outputdirectories):
        random_folder_name = binascii.hexlify(os.urandom(16)).decode('ascii')
        os.mkdir(os.path.join(gis_env['GISDBASE'], str(gis_env['LOCATION_NAME']), random_folder_name))
        shutil.copytree(os.path.join(gis_env['GISDBASE'], str(gis_env['LOCATION_NAME']), "PERMANENT"),
                        os.path.join(gis_env['GISDBASE'], str(gis_env['LOCATION_NAME']), random_folder_name,
                                     "PERMANENT"))
        bash_location = os.path.join(gis_env['GISDBASE'], str(gis_env['LOCATION_NAME']), random_folder_name)
        outputdir = os.path.join(gis_env['GISDBASE'], str(gis_env['LOCATION_NAME']), random_folder_name,
                                 "PERMANENT")
        outputdirectories.append(
            os.path.join(gis_env['GISDBASE'], str(gis_env['LOCATION_NAME']), random_folder_name))
        # 1. select barriers from selected spatial units (sub-watersheds)
        indizes_protected_areas = tuple(np.where(solution == True)[0])


        # names used in GRASS GIS functions
        dem_ras = "demraster"
        landuse_ras = "landuse_ras"
        rfactor_rasters = ["rfactor_ras" + str(i) for i in selected_realization_ids]
        kfactor_rasters = ["kfactor_ras" + str(i) for i in selected_realization_ids]
        rusle_rasters = ["rusle_ras" + str(i) for i in selected_realization_ids]
        # also load raster names
        input_r_factors = []
        for filename in os.listdir(input_r_factor_dir):
            if filename.endswith(".tif"):
                input_r_factors.append(os.path.join(input_r_factor_dir, filename))
        input_k_factors = []
        for filename in os.listdir(input_k_factor_dir):
            if filename.endswith(".tif"):
                input_k_factors.append(os.path.join(input_k_factor_dir, filename))

        cfactor_ras = "cfactor_ras"
        slope_degrees = "slope_degrees_ras"
        output_flowline_ras = "flowlineraster"
        output_flowlength_ras = "flowlengthraster"
        output_flowacc_ras = "flowaccraster"
        watersheds_ras = "watersheds_ras"
        protected_watersheds_ras = "protected_watersheds_ras"
        m_ras = "m_ras"
        s_factor_ras = "sfactor_ras"
        p_factor_ras = "pfactor_ras"
        lfactor_ras = "lfactor_ras"
        lsfactor_ras = "lsfactor_ras"
        contourlines_ras = "contourlines_ras"
        selected_contour_lines_ras = "selected_contour_lines_ras"
        csv_stats = [os.path.join(gis_env['GISDBASE'], str(gis_env['LOCATION_NAME']), random_folder_name,
                                  "rusle_total_{}.csv".format(str(i))) for i in selected_realization_ids]
        pi_constant = math.pi
        with open(os.path.join(bash_location, "rusle.sh"), "w") as f:
            f.write("#!/bin/bash" + "\n")
            f.write('g.proj -c epsg=32637' + "\n")
            f.write(r'r.in.gdal input={} band=1 output={} --overwrite -o'.format(input_dem, dem_ras) + "\n")
            f.write(r'r.in.gdal input={} band=1 output={} --overwrite -o'.format(input_landuse_raster,
                                                                                 landuse_ras) + "\n")
            f.write(r'r.in.gdal input={} band=1 output={} --overwrite -o'.format(input_c_factor,
                                                                                 cfactor_ras) + "\n")
            f.write(r'r.in.gdal input={} band=1 output={} --overwrite -o'.format(input_slope_degrees,
                                                                                 slope_degrees) + "\n")
            # here we need loops
            [f.write(r'r.in.gdal input={} band=1 output={} --overwrite -o'.format(input_r_factors[i].replace("\\", "/"),
                                                                                  rfactor_rasters[
                                                                                      i]) + "\n") \
             for i in range(len(selected_realization_ids))]

            [f.write(r'r.in.gdal input={} band=1 output={} --overwrite -o'.format(input_k_factors[i].replace("\\", "/"),
                                                                                  kfactor_rasters[
                                                                                      i]) + "\n") \
             for i in range(len(selected_realization_ids))]

            f.write(r'r.in.gdal input={} band=1 output={} --overwrite -o'.format(input_contour_lines,
                                                                                 contourlines_ras) + "\n")
            f.write(r'r.in.gdal input={} band=1 output={} --overwrite -o'.format(input_watersheds, watersheds_ras) + "\n")

            f.write(
                r'g.region raster={}'.format(dem_ras)+ "\n")

            if len(indizes_protected_areas) == 0:
                protected_watersheds_ras  = None
            else:
                formula_insert = ""
                for i in range(1,len(indizes_protected_areas)):
                    formula_insert += " || " + """ "{watersheds_ras}" == {id}""".format(id=indizes_protected_areas[i]+1,
                                                                                watersheds_ras=watersheds_ras)
                formula = """ if(("{watersheds_ras}" == {id0} {formula_insert}  ),1,0)""".format(
                        id0=indizes_protected_areas[0]+1, formula_insert=formula_insert,watersheds_ras=watersheds_ras, protected_watersheds= protected_watersheds_ras)
                f.write(
                r'r.mapcalc --overwrite expression=""{watersheds_ras}" = {formula}"'.format(
                    watersheds_ras=protected_watersheds_ras,
                    formula =formula) + "\n")

            if len(indizes_protected_areas) == 0:
                selected_contour_lines_ras = None
            else:
                formula_insert = ""
                for i in range(1, len(indizes_protected_areas)):
                    formula_insert = " || " + """ "{contours}" == {id}""".format(id=indizes_protected_areas[i]+1,
                                                                                contours = contourlines_ras)
                formula = """ if(("{contours}" == {id0} {formula_insert}  ),1,0)""".format(contours = contourlines_ras,
                    id0=indizes_protected_areas[0]+1, formula_insert=formula_insert,
                    selected_contours = selected_contour_lines_ras)
                f.write(
                    r'r.mapcalc --overwrite expression=""{selected_contours}" = {formula}"'.format(
                        selected_contours = selected_contour_lines_ras,
                        formula=formula) + "\n")
            if len(indizes_protected_areas) > 0:
                f.write(
                        r'r.flow  elevation={} barrier={} flowline={} flowlength={} flowaccumulation={} --overwrite'.format(
                            dem_ras, selected_contour_lines_ras, output_flowline_ras, output_flowlength_ras,
                            output_flowacc_ras) + "\n")
            else:
                f.write(
                    r'r.flow  elevation={} flowline={} flowlength={} flowaccumulation={} --overwrite'.format(
                        dem_ras, output_flowline_ras, output_flowlength_ras,
                        output_flowacc_ras) + "\n")

            f.write(r'g.region raster={}'.format(output_flowlength_ras)+ "\n")
            if len(indizes_protected_areas)>0:
                f.write(
                    r'r.mapcalc --overwrite expression=""{pfactor_ras}" = if({landuse_ras} == 1, (if({protected_watersheds_ras}==1, (if({slope}>15, 0.356 , 0.325)) , 1)),(if({landuse_ras} == 2, if(!isnull({protected_watersheds_ras}), 0.7 , 1), (if({landuse_ras} == 3 || {landuse_ras} == 4 || {landuse_ras} == 5, 1,0)) )) )"'.format(
                        pfactor_ras=p_factor_ras, landuse_ras=landuse_ras,
                        protected_watersheds_ras=protected_watersheds_ras,slope= slope_degrees) + "\n")
            else:
                f.write(
                    r'r.mapcalc --overwrite expression=""{pfactor_ras}" = if({landuse_ras} == 1, 1,(if({landuse_ras} == 2, 1, (if({landuse_ras} == 3 || {landuse_ras} == 4 || {landuse_ras} == 5, 1,0)) )) )"'.format(
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
            [f.write(
                r'r.mapcalc --overwrite expression=""{rusle}"={r}*{k}*{ls}*{c}*{p}"'.format(
                    r=rfactor_rasters[i],
                    k=kfactor_rasters[i],
                    ls=lsfactor_ras,
                    c=cfactor_ras,
                    p=p_factor_ras,
                    rusle=rusle_rasters[i]) + "\n") \
                for i in range(len(selected_realization_ids))]

            [f.write(r'r.univar -t map={rusle} separator=comma output="{stats}" --overwrite'.format(
                rusle=rusle_rasters[i], stats=csv_stats[i]) + "\n") for i in range(len(selected_realization_ids))]

            if savemaps == True:
                [f.write(
                    r'r.out.gdal -t -m input={} output="{}/rusle_{}.tif" format=GTiff  createopt=TFW=YES,COMPRESS=LZW --overwrite'.format(
                        rusle_rasters[i], outputdir,i) + "\n") for i in range(len(selected_realization_ids))]
                f.write(
                    r'r.out.gdal -t -m input={} output="{}/ls_factor.tif" format=GTiff  createopt=TFW=YES,COMPRESS=LZW --overwrite'.format(
                        lsfactor_ras, outputdir) + "\n")
                f.write(
                    r'r.out.gdal -t -m input={} output="{}/c_factor.tif" format=GTiff  createopt=TFW=YES,COMPRESS=LZW --overwrite'.format(
                        cfactor_ras, outputdir) + "\n")
                f.write(
                    r'r.out.gdal -t -m input={} output="{}/p_factor.tif" format=GTiff  createopt=TFW=YES,COMPRESS=LZW --overwrite'.format(
                        p_factor_ras, outputdir) + "\n")
                f.write(
                    r'r.out.gdal -t -m input={} output="{}/selected_barriers.tif" format=GTiff  createopt=TFW=YES,COMPRESS=LZW --overwrite'.format(
                        selected_contour_lines_ras, outputdir) + "\n")
                f.write(
                    r'r.out.gdal -t -m input={} output="{}/selected_watersheds.tif" format=GTiff  createopt=TFW=YES,COMPRESS=LZW --overwrite'.format(
                        protected_watersheds_ras, outputdir) + "\n")

                #also save in shapefile format for nicer visual outputs
                f.write(
                    r'v.in.ogr min_area=0.0 snap=-1.0 input={} layer="contour_lines_with_watershed_id_enerata" output="contourlines_shp" --overwrite -o'.format(
                        contourlines_shp) + "\n")
                f.write(
                    r'v.in.ogr min_area=0.0 snap=-1.0 input={} layer="Basins_4" output={} --overwrite -o'.format(
                        watersheds_shp, "watersheds_shp") + "\n")
                f.write(
                    r'v.extract input=contourlines_shp  output=selected_contour_lines_shp layer="contour_lines_with_watershed_id_enerata" where="pos_rank in {}"'.format(
                        indizes_protected_areas) + "\n")

                f.write(
                    r'v.extract input={}  output={} layer="Basins_4" where="pos_rank in {}"'.format(
                        "watersheds_shp", "selected_watersheds_shp", indizes_protected_areas) + "\n")

                f.write(
                    r'v.out.ogr input={} output="{}/terraces.geojson" format=GeoJSON  --overwrite'.format(
                        "selected_contour_lines_shp", outputdir) + "\n")

                f.write(
                    r'v.out.ogr input={} output="{}/protected_watersheds.geojson" format=GeoJSON --overwrite'.format(
                        "selected_watersheds_shp", outputdir) + "\n")

        f.close()

        grass7bin_win = r'C:\OSGeo4W64\bin\grass78.bat'
        startcmd = [grass7bin_win, outputdir, '--exec', os.path.join(bash_location, "rusle.sh")]
        p = subprocess.Popen(startcmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        processes.append(p)
        return

    start = time.time()
    processes = []
    outputdirectories = []

    for solution in solutions:
        compute_rusle_per_solution_under_uncertainty(solution, processes, outputdirectories)

    begintotal = time.time()
    output = [p.wait() for p in processes]
    endtotal = time.time()
    f1 = []

    print(outputdirectories)
    for dir in outputdirectories:
        obj_values = []
        for i in selected_realization_ids:
            try:
                df = pd.read_csv(os.path.join(dir, 'rusle_total_{}.csv'.format(str(i))))
                obj_values.append(df['sum'].iloc[0]/study_area_ha)
            except:
                obj_values.append(sys.maxsize)
        f1.append(obj_values)
    end = time.time()
    return f1, end - start, outputdirectories


def calculate_labour_requirements_under_uncertainty(solutions, watersheds_raster_dataset, labour_requirement_datasets,
                                                    selected_realization_ids,study_area_ha):
    def compute_labour_requirements_per_solution_under_uncertainty(solution):
        tempdir = tempfile.TemporaryDirectory()
        outputdir = tempdir.name

        indizes_protected_areas = tuple(np.where(solution == True)[0])

        active_soil_conservation_dataset = np.isin(watersheds_raster_dataset, indizes_protected_areas)

        f2 = []
        for i in range(len(selected_realization_ids)):
            sum_required_labour = np.sum(labour_requirement_datasets[i][[active_soil_conservation_dataset == True]])
            f2.append(sum_required_labour/study_area_ha)
        del active_soil_conservation_dataset
        return f2

    start = time.time()

    f2 = [compute_labour_requirements_per_solution_under_uncertainty(solution) for solution in solutions]
    end = time.time()
    return f2, end - start

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

        def __init__(self, m=1,nr_genes = 100,save_maps = False, normalize=False, **kwargs):
            self.m = m
            self.nr_genes = nr_genes
            self.normalize = normalize
            self.save_maps = save_maps
            super().__init__(n_var=nr_genes, **kwargs)

        def _calc_pareto_front(self, n_pareto_points=100):
            x = 1 + anp.linspace(0, 1, n_pareto_points) * 30
            pf = anp.column_stack([x, (self.m - 1) / x])
            if self.normalize:
                pf = normalize(pf)
            return pf

        def _evaluate(self, X, out, *args, **kwargs):
            generation_id = 0
            solutions = [X[:, :self.nr_genes]][0]
            print(solutions)

            #input data for enerata
            # wrkdir = r"C:/Users/morit/AppData/Local/Temp/grassdata/mytemploc_utm32n2"
            # input_dem = wrkdir + "//" + 'DEM_Enerata_filled.tif'
            # extent = compute_extent(input_dem)
            # crs = compute_crs(input_dem)
            # input_contour_lines_raster = wrkdir + "//" + "contour_lines_with_watershed_id.tif"
            # input_landuse_raster = wrkdir + "//" + "landuse_raster.tif"
            # input_r_factors = wrkdir + "//" + "r_factor_realizations"
            # input_k_factors = wrkdir + "//" + "k_factor_realizations"
            # input_c_factor = wrkdir + "//" + "c_factor_clipped.tif"
            # input_slope_degrees = wrkdir + "//" + "Slope_degrees.tif"
            # labour_requirements_dir = wrkdir + "//" + "labor_requirements"
            # # load only dataset of watersheds raster to accelerate computation
            # input_watersheds_raster = wrkdir + "//" + "watersheds_raster_with_id.tif"
            # # load dataset of watersheds raster to accelerate computation
            # input_watersheds_dataset = load_raster_dataset(input_watersheds_raster)
            # input_watersheds_shp = wrkdir + "//" + "Basins_4.shp"
            # input_contour_lines_shp = wrkdir + "//" + "contour_lines_with_watershed_id_enerata.shp"

            #input data gumobila
            wrkdir = r"C:/Users/morit/AppData/Local/Temp/grassdata/gumobila"
            input_dem = wrkdir + "//" + 'DEM_Gumobila2.tif'
            extent = compute_extent(input_dem)
            crs = compute_crs(input_dem)
            input_contour_lines_raster = wrkdir + "//" + "contourlines_raster_with_id.tif"
            input_landuse_raster = wrkdir + "//" + "landuse_raster.tif"
            input_r_factors = wrkdir + "//" + "r_factor_realizations"
            input_k_factors = wrkdir + "//" + "k_factor_realizations"
            input_c_factor = wrkdir + "//" + "c_factor_clipped.tif"
            input_slope_degrees = wrkdir + "//" + "Slope_degrees.tif"
            labour_requirements_dir = wrkdir + "//" + "labor_requirements"
            # load only dataset of watersheds raster to accelerate computation
            input_watersheds_raster = wrkdir + "//" + "watersheds_raster_with_id.tif"
            # load dataset of watersheds raster to accelerate computation
            input_watersheds_dataset = load_raster_dataset(input_watersheds_raster)
            input_watersheds_shp = wrkdir + "//" + "watersheds_with_id.shp"
            input_contour_lines_shp = wrkdir + "//" + "contour_lines_with_watershed_id.shp"




            selected_realization_ids = [i for i in range(22)]
            study_area_ha = get_area_in_ha(input_watersheds_shp)
            # load dataset of labor requirement raster to accelerate computation
            input_labour_requirements = []
            for filename in os.listdir(labour_requirements_dir):
                if filename.endswith(".tif"):
                    input_labour_requirements.append(load_raster_dataset(os.path.join(labour_requirements_dir, filename)))

            f1, timef1, outputdirectories = calculate_rusle_under_uncertainty(solutions, input_dem,
                                                                              input_landuse_raster, input_watersheds_raster,
                                                                              input_contour_lines_raster, input_slope_degrees,
                                                                              input_r_factors,
                                                                              input_k_factors, input_c_factor,
                                                                              selected_realization_ids,input_watersheds_shp, input_contour_lines_shp,
                                                                              extent, crs,study_area_ha=study_area_ha,savemaps=self.save_maps)
            f2, timef2 = calculate_labour_requirements_under_uncertainty(solutions, input_watersheds_dataset,
                                                                      input_labour_requirements,selected_realization_ids,study_area_ha)
            if self.save_maps is False:
                for dir in outputdirectories:
                    shutil.rmtree(dir, ignore_errors=False, onerror=None)

            print("Generation: " + str(generation_id))
            generation_id += 1
            print("Execution time objective function 1: " + str(timef1))
            print("Execution time objective function 2: " + str(timef2))
            if self.normalize:
                f1 = normalize(f1, 1, self.n + 1)
                f2 = normalize(f2, 1, self.n + 1)
            print("obj1:")
            print(f1)
            print("obj2:")
            print(f2)

            population_with_reshaped_objectivevalues = []
            for i in range(len(f1)):
                population_with_reshaped_objectivevalues.append([f1[i],f2[i]])
            out["F"] = population_with_reshaped_objectivevalues
            print("Population")
            print(out["F"])

    nr_watersheds= get_nr_watersheds(input_watersheds_shp = input_watersheds_shp)
    problem = MyProblem(parallelization=True, nr_genes = nr_watersheds, save_maps=False)

    algorithm = NSGA2Unc(
        pop_size=50,
        sampling=get_sampling("bin_random"),
        crossover=get_crossover("bin_hux"),
        mutation=get_mutation("bin_bitflip"),
        selection=get_selection("random"),
        eliminate_duplicates=True)

    res = minimize(problem,
                   algorithm,
                   ('n_gen', 100),
                   #verbose=True,
                   #pf=problem.pareto_front(use_cache=False),
                   save_history=True)
    #
    # """
    # res.X design space values are
    # res.F objective spaces values
    # res.G constraint values
    # res.CV aggregated constraint violation
    # res.algorithm algorithm object
    # res.pop final population object
    # res.history history of algorithm object. (only if save_history has been enabled during the algorithm initialization)
    # res.time the time required to run the algorithm
    # """
    #
    import matplotlib.pyplot as plt
    #
    # # iterate over the deepcopies of algorithms
    outputdir = r"C:\Users\morit\OneDrive - Universität Münster\PhD\Kooperation_GIZ\Data\Optimization_Gumobila\optimization_result\pareto_fronts"
    populations = []
    for algorithm in res.history:
        # retrieve the optimum from the algorithm
        opt = algorithm.opt
        _F = opt.get("F")
        _X = opt.get("X")
        populations.append([_F, _X])
    #
    #
    with open(os.path.join(outputdir, 'all_populations_gumobila.pkl'), 'wb') as handle:
        pickle.dump(populations, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open(os.path.join(outputdir, 'all_populations2.pkl'), 'rb') as handle:
    #     populations = pickle.load(handle)

    # i=0
    # obj2_min = 210000
    # obj2_max = 400000
    # obj1_min = 30000
    # obj1_max = 59000
    #
    # for pop in populations:
    #     # retrieve the optimum from the algorithm
    #     stochastically_non_dominated_solutions, sorted_expected_fitnesses_of_dominated_solutions = ranking_and_fitness_assignment_SPGA(
    #         pop, n_survive=10)
    #     plot_pareto_front(pop, stochastically_non_dominated_solutions,
    #                       sorted_expected_fitnesses_of_dominated_solutions, population_size=10, generation_id=i,
    #                       axis_ranges=[obj1_min, obj1_max, obj2_min, obj2_max],
    #                       outputdir=outputdir)
    #     i+=1




