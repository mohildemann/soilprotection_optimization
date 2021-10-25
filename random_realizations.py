import math
import numpy as np
import scipy.stats as st

def moving_average_realization(mean_raster, interquantile_distance,nr_realizations, filter_range, NAN_value,lower_cutoff_value = None, upper_cutoff_value = None):
    NAN_positions = interquantile_distance== NAN_value
    # input from ISRIC are layers with 90% quantile prediction intervals that need to be converted to std.
    # Prediction intervals were computed from 10-fold cross-validation
    # z-score of 90% prediction interval from t-distribution with 9 degrees of freedom is 1.833. Needed due to small sample size of 10
    # if different data is used, this first part of the code needs o be adapted
    z = 1.833
    # number of obersevations = 10 from 10-fold cross-validation
    N = 10
    # SD = √N * (upper limit - lower limit) / (2*z)
    interquantile_distance = interquantile_distance.astype(float)
    std = math.sqrt(N) * (interquantile_distance) / (2*z)
    std[NAN_positions] = NAN_value

    # test for first cell if intervals from realizations are coherent to 90% prediction intervals from ISRIC
    first_cell = np.random.normal(mean_raster[0][0], std[0][0], nr_realizations)
    test_intervals_first_cell = st.t.interval(alpha=0.9, df=10-1, loc=np.mean(first_cell), scale=st.sem(first_cell))

    realizations_array = np.array([[[0 for k in range(nr_realizations)] for j in range(mean_raster.shape[1])] for i in range(mean_raster.shape[0])])
    for i in range(mean_raster.shape[0]):
        for j in range(mean_raster.shape[1]):
            if mean_raster[i][j] == NAN_value or interquantile_distance[i][j] == NAN_value:
                realizations_array[i][j] = NAN_value
            else:
                drawn_values = np.random.normal(mean_raster[i][j], std[i][j], nr_realizations)
                # check for cutoff-values
                if lower_cutoff_value is not None:
                    drawn_values[drawn_values < lower_cutoff_value] = lower_cutoff_value
                elif upper_cutoff_value is not None:
                    drawn_values[drawn_values < upper_cutoff_value] = upper_cutoff_value
                realizations_array[i][j] = np.random.normal(mean_raster[i][j], std[i][j], nr_realizations)




    #moving window average
    moving_averages = realizations_array.astype(float).copy()

    #array to check if border values were excluded correctly
    border_count = 0
    border_array_illustration = np.zeros((mean_raster.shape[0],mean_raster.shape[1]))

    for realization_id in range(nr_realizations):
        for i in range(mean_raster.shape[0]):
            for j in range(mean_raster.shape[1]):
                nr_neighbors = 0
                sum_of_neighbors = 0
                for k in range(filter_range+1):
                    for l in range(filter_range+1):
                        #check for borders of raste
                        if k == 0 and l == 0:
                            if realizations_array[i][j][realization_id] != NAN_value:
                                nr_neighbors += 1
                                sum_of_neighbors += realizations_array[i][j][realization_id]
                        elif i - k >= 0 and i + k < mean_raster.shape[0] and j - l >=0 and j + l < mean_raster.shape[1]:
                            if realizations_array[i-k][j-l][realization_id] != NAN_value:
                                nr_neighbors += 1
                                sum_of_neighbors += realizations_array[i-k][j-l][realization_id]
                            if realizations_array[i+k][j+l][realization_id] != NAN_value:
                                nr_neighbors += 1
                                sum_of_neighbors += realizations_array[i+k][j+l][realization_id]
                        else:
                            border_count += 1
                            border_array_illustration[i,j] = 1
                if nr_neighbors>0:
                    average = sum_of_neighbors / nr_neighbors
                    moving_averages[i][j][realization_id] = average
                else:
                    moving_averages[i][j][realization_id] = NAN_value
    rescaled_moving_averages = []
    for i in range(nr_realizations):
        x = moving_averages[:,:,i]
        m1 = np.mean(x[~NAN_positions])
        s1 = np.std(x[~NAN_positions])
        m2 = np.mean(mean_raster[~NAN_positions])
        s2 = np.mean(std[~NAN_positions])
        y = m2 +(x-m1) * (s2/s1)
        y[NAN_positions] = NAN_value
        rescaled_moving_averages.append(y)
    rescaled_moving_averages = np.array(rescaled_moving_averages)
    #test, whether the correct percentage of the data lies in the defined intervals
    percentage_out_of_percentiles = np.count_nonzero((rescaled_moving_averages[0] > mean_raster + interquantile_distance) | (rescaled_moving_averages[0] < mean_raster - interquantile_distance))/(mean_raster.shape[0] * mean_raster.shape[1])
    return rescaled_moving_averages


def percentage_based_group_realization(silt,sand,clay, nr_realizations, NA_value):
    #function needed to simulate sand,silt and clay content that need to sum up to 100%
    NA_cells = silt[0].astype(int) == NA_value
    sum_surface = []
    for i in range(nr_realizations):
        #the soil data is available in g/kg, therefore we need to divide by 10 to get percentages
        silt_percent  = silt[i] / 10
        sand_percent = sand[i]/ 10
        clay_percent = clay[i]/ 10
        silt_percent[NA_cells] = NA_value
        sand_percent[NA_cells] = NA_value
        clay_percent[NA_cells] = NA_value
        summed_up = silt_percent + sand_percent + clay_percent
        summed_up[NA_cells] = NA_value
        compensation_factor = 100/summed_up
        compensation_factor[NA_cells] = NA_value
        #now we need to rescale stepwise in order to maintain possible values between 0 and 100
        silt[i][~NA_cells] = silt_percent[~NA_cells] * (compensation_factor[~NA_cells])
        sand[i][~NA_cells] = sand_percent[~NA_cells] * (compensation_factor[~NA_cells])
        clay[i][~NA_cells] = clay_percent[~NA_cells] * (compensation_factor[~NA_cells])

    return silt,sand,clay


