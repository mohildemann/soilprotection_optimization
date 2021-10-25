import numpy as np
import scipy.stats as st
from scipy import special as sp
import math
import pandas as pd
import matplotlib.pyplot as plt
import os, os.path

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.docs import parse_doc_string
from pymoo.model.individual import Individual
from pymoo.model.survival import Survival
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import FloatRandomSampling
from pymoo.operators.selection.tournament_selection import compare, TournamentSelection
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.dominator import Dominator
from pymoo.util.misc import find_duplicates, has_feasible
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort

# =========================================================================================================
# Implementation
# =========================================================================================================

def binary_tournament(pop, P, algorithm, **kwargs):
    if P.shape[1] != 2:
        raise ValueError("Only implemented for binary tournament!")

    tournament_type = algorithm.tournament_type
    S = np.full(P.shape[0], np.nan)

    for i in range(P.shape[0]):

        a, b = P[i, 0], P[i, 1]

        # if at least one solution is infeasible
        if pop[a].CV > 0.0 or pop[b].CV > 0.0:
            S[i] = compare(a, pop[a].CV, b, pop[b].CV, method='smaller_is_better', return_random_if_equal=True)

        # both solutions are feasible
        else:

            if tournament_type == 'comp_by_dom_and_crowding':
                rel = Dominator.get_relation(pop[a].F, pop[b].F)
                if rel == 1:
                    S[i] = a
                elif rel == -1:
                    S[i] = b

            elif tournament_type == 'comp_by_rank_and_crowding':
                S[i] = compare(a, pop[a].get("rank"), b, pop[b].get("rank"),
                               method='smaller_is_better')

            else:
                raise Exception("Unknown tournament type.")

            # if rank or domination relation didn't make a decision compare by crowding
            if np.isnan(S[i]):
                S[i] = compare(a, pop[a].get("crowding"), b, pop[b].get("crowding"),
                               method='larger_is_better', return_random_if_equal=True)

    return S[:, None].astype(int, copy=False)

class NSGA2Unc(GeneticAlgorithm):

    def __init__(self,
                 pop_size=100,
                 sampling=FloatRandomSampling(),
                 selection=TournamentSelection(func_comp=binary_tournament),
                 crossover=SimulatedBinaryCrossover(eta=15, prob=0.9),
                 mutation=PolynomialMutation(prob=None, eta=20),
                 eliminate_duplicates=True,
                 n_offsprings=None,
                 display=MultiObjectiveDisplay(),
                 **kwargs):
        """

        Parameters
        ----------
        pop_size : {pop_size}
        sampling : {sampling}
        selection : {selection}
        crossover : {crossover}
        mutation : {mutation}
        eliminate_duplicates : {eliminate_duplicates}
        n_offsprings : {n_offsprings}

        """

        super().__init__(pop_size=pop_size,
                         sampling=sampling,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         survival=ProbabilisticDominanceSurvival(),
                         eliminate_duplicates=eliminate_duplicates,
                         n_offsprings=n_offsprings,
                         display=display,
                         **kwargs)

        self.tournament_type = 'comp_by_dom_and_crowding'

    def _set_optimum(self, **kwargs):
        pop_uncertain_obj_values = [sol.F for sol in self.pop]
        means, stds, intervals = compute_stats_of_uncertain_objective_values(pop_uncertain_obj_values)
        # Step 1: identify all stochastically non-dominated solutions (rank 1)
        # Stochastic dominance is  evaluated by using the sample average
        nds = NonDominatedSorting()
        ranked_on_sample_means = nds.do(np.array(means), return_rank=True)
        stochastically_non_dominated = ranked_on_sample_means[0][0]
        self.opt = self.pop[stochastically_non_dominated]

# ---------------------------------------------------------------------------------------------------------
# Survival Selection
# ---------------------------------------------------------------------------------------------------------


class ProbabilisticDominanceSurvival(Survival):

    def __init__(self) -> None:
        super().__init__(filter_infeasible=True)
        #self.nds = NonDominatedSorting()

    def _do(self, problem, pop, n_survive, D=None, **kwargs):

        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=False)

        # the final indices of surviving individuals
        survivors = []
        stochastically_non_dominated_solutions, sorted_expected_fitnesses_of_dominated_solutions = ranking_and_fitness_assignment_SPGA(F, n_survive)
        [survivors.append(i) for i in stochastically_non_dominated_solutions]

        for i in range(n_survive-len(survivors)):
            survivors.append(int(sorted_expected_fitnesses_of_dominated_solutions[i][0]))
        return pop[survivors]


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



parse_doc_string(NSGA2Unc.__init__)
