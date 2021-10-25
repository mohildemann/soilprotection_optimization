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

import pymoo
from pymoo.factory import get_problem, get_reference_directions, get_visualization
from pymoo.util.plotting import plot
from pymoo.util.nds import fast_non_dominated_sort
from pymoo.util.dominator import Dominator
from pymoo.util.function_loader import load_function
from pymoo.model.selection import Selection

#use class from pymoo
class NonDominatedSorting:

    def __init__(self, epsilon=None, method="fast_non_dominated_sort") -> None:
        super().__init__()
        self.epsilon = epsilon
        self.method = method

    def do(self, F, return_rank=False, only_non_dominated_front=False, n_stop_if_ranked=None, **kwargs):
        def find_non_dominated(F, _F=None):
            M = Dominator.calc_domination_matrix(F, _F)
            I = np.where(np.all(M >= 0, axis=1))[0]
            return I

        def rank_from_fronts(fronts, n):
            # create the rank array and set values
            rank = np.full(n, 1e16, dtype=int)
            for i, front in enumerate(fronts):
                rank[front] = i

            return rank

        F = F.astype(float)

        # if not set just set it to a very large values because the cython algorithms do not take None
        if n_stop_if_ranked is None:
            n_stop_if_ranked = int(1e8)
        func = load_function(self.method)

        # set the epsilon if it should be set
        if self.epsilon is not None:
            kwargs["epsilon"] = float(self.epsilon)

        fronts = func(F, **kwargs)

        # convert to numpy array for each front and filter by n_stop_if_ranked if desired
        _fronts = []
        n_ranked = 0
        for front in fronts:

            _fronts.append(np.array(front, dtype=int))

            # increment the n_ranked solution counter
            n_ranked += len(front)

            # stop if more than this solutions are n_ranked
            if n_ranked >= n_stop_if_ranked:
                break

        fronts = _fronts

        if only_non_dominated_front:
            return fronts[0]

        if return_rank:
            rank = rank_from_fronts(fronts, F.shape[0])
            return fronts, rank

        return fronts

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
def ranking_and_fitness_assignment_SPGA(uncertain_pf):
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
    open_spots = 20 - len(stochastically_non_dominated)
    # identify solutions that have highest fitness for open spots

    expected_fitnesses_numpy = np.array(expected_fitnesses)
    sorted_expected_fitnesses_of_dominated_solutions = expected_fitnesses_numpy[expected_fitnesses_numpy[:, 1].argsort()][::-1]
    sorted_expected_fitnesses_of_dominated_solutions[:,0] = sorted_expected_fitnesses_of_dominated_solutions[:,0].astype(int)
    return stochastically_non_dominated, sorted_expected_fitnesses_of_dominated_solutions

class ProbabilisticDominanceSelection(Selection):

    def _do(self, Hm, n_select, n_parents=2, **kwargs):
        n_pop = len(Hm) // 2

        _, rank = NonDominatedSorting().do(Hm.get('F'), return_rank=True)

        Pc = (rank[:n_pop] == 0).sum() / len(Hm)
        Pd = (rank[n_pop:] == 0).sum() / len(Hm)

        # number of random individuals needed
        n_random = n_select * n_parents * self.pressure
        n_perms = math.ceil(n_random / n_pop)
        # get random permutations and reshape them
        P = random_permuations(n_perms, n_pop)[:n_random]
        P = np.reshape(P, (n_select * n_parents, self.pressure))
        if Pc <= Pd:
            # Choose from DA
            P[::n_parents, :] += n_pop
        pf = np.random.random(n_select)
        P[1::n_parents, :][pf >= Pc] += n_pop

        # compare using tournament function
        S = self.f_comp(Hm, P, **kwargs)

        return np.reshape(S, (n_select, n_parents))
