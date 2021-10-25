import pandas as pd
import dash
#import visualization
import dash_bootstrap_components as dbc
import numpy as np
import postprocessing_optimization
#import visualization
class Solution:
    _id = 0

    def __init__(self, representation, objective_values):
        self._solution_id = Solution._id
        Solution._id += 1
        self.representation = representation
        self.objective_values = objective_values
        #self.metadata = metadata

"""
res.X design space values are
res.F objective spaces values
"""
import os
import pickle
outputdir = r"C:/Users/morit/OneDrive - Universität Münster/PhD/Kooperation_GIZ/Data/Optimization_Enerata/Output Enerata/pareto_fronts/test"
with open(os.path.join(outputdir, 'all_populations6.pkl'), 'rb') as handle:
        populations = pickle.load(handle)

final_population =  populations[-1]
final_population_objective_values = [F for F in final_population[0]]
final_population_genes = [X for X in final_population[1]]
#final_population_metadata = [elem.data for elem in final_population_df[0]]
optimal_solutions = []
for i in range(len(final_population_objective_values)):
    optimal_solutions.append(Solution(final_population_genes[i],final_population_objective_values[i]))
print()
#copy of inputs in C:\Users\morit\OneDrive - Universität Münster\PhD\Kooperation_GIZ\Data\Optimization_Enerata\copy_grass_gis_env

input_dem = r'C:/Users/morit/AppData/Local/Temp/grassdata/mytemploc_utm32n2/DEM_Enerata_filled.tif'
input_contour_lines = r'C:/Users/morit/AppData/Local/Temp/grassdata/mytemploc_utm32n2/contour_lines_with_watershed_id_enerata.shp'
input_watersheds = r'C:/Users/morit/AppData/Local/Temp/grassdata/mytemploc_utm32n2/Basins_4.shp'
input_landuse_raster = r"C:/Users/morit/AppData/Local/Temp/grassdata/mytemploc_utm32n2/landuse_raster.tif"
input_r_factor = r"C:/Users/morit/AppData/Local/Temp/grassdata/mytemploc_utm32n2/r_factor_realizations/r_factor_0_clipped.tif"
input_k_factor = r"C:/Users/morit/AppData/Local/Temp/grassdata/mytemploc_utm32n2/k_factor_realizations/k_factor_r_0clipped.tif"
input_c_factor = r"C:/Users/morit/AppData/Local/Temp/grassdata/mytemploc_utm32n2/c_factor_clipped.tif"
input_slope_degrees = r"C:/Users/morit/AppData/Local/Temp/grassdata/mytemploc_utm32n2/Slope_degrees.tif"
labour_requirements = r"C:/Users/morit/AppData/Local/Temp/grassdata/mytemploc_utm32n2/labor_requirements/labour_requirements_0.tif"
output = postprocessing_optimization.calculate_rusle(optimal_solutions, input_dem, input_landuse_raster, input_watersheds,
                                               input_contour_lines, input_slope_degrees, input_r_factor,
                                               input_k_factor, input_c_factor, savemaps = True)

# output = r"C:/Users/morit/AppData/Local/Temp/grassdata/32246e58df0d3191e905537d95454bdc"
#
# css = [dbc.themes.BOOTSTRAP]
#
# app = dash.Dash(
#     __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}],external_stylesheets = [dbc.themes.BOOTSTRAP]
# )
#
# server = app.server
# app.layout = visualization.create_layout(app)
#
# visualization.demo_callbacks(app, optimal_solutions, output)
#
# # Running server
# app.run_server(debug=True)