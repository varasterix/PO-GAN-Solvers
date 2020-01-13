import sys
from src.database.databaseTools import compute_tsp_heuristic_solution

nb_cities = int(sys.argv[1])
instance_id = int(sys.argv[2])
tsp_file_path = sys.argv[3]
time_limit = int(sys.argv[4])

# Execution : compute heuristic and create TSP heuristic solution file
compute_tsp_heuristic_solution(nb_cities, instance_id, tsp_file_path, time_limit)
