import os
import src.constants as constants
from src.database.databaseTools import compute_tsp_nnh_solution_from_choco_database, \
    compute_tsp_nnh_two_opt_solution_from_choco_database


time_limit = 10
choco_path = constants.PARAMETER_TSP_CHOCO_DATA_FILES
nnh_path = constants.PARAMETER_TSP_NNH_DATA_FILES
nnh_two_opt_path = constants.PARAMETER_TSP_NNH_TWO_OPT_DATA_FILES


if __name__ == "__main__":
    for dataSet in os.listdir(choco_path):
        details = dataSet.split('.')[0].split('_')
        nb_cities, instance_id = int(details[1]), int(details[2])
        compute_tsp_nnh_solution_from_choco_database(nb_cities, instance_id, nnh_path, choco_path)
        compute_tsp_nnh_two_opt_solution_from_choco_database(nb_cities, instance_id, path=nnh_two_opt_path,
                                                             choco_path=choco_path, time_limit=time_limit)
