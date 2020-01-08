import sys
from src.database.databaseTools import generate_tsp_file

nb_cities = int(sys.argv[1])
instance_id = int(sys.argv[2])
tsp_file_path = sys.argv[3]
highest_weight = int(sys.argv[4])

# Execution of the generator
generate_tsp_file(nb_cities, instance_id, tsp_file_path, highest_weight)
