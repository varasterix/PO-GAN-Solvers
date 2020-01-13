import sys
from src.database.databaseTools import generate_tsp_file

nb_cities = int(sys.argv[1])
instance_id = int(sys.argv[2])
tsp_file_path = sys.argv[3]
highest_weight = int(sys.argv[4])
symmetric_instance = bool(int(sys.argv[5]))

# Execution of the generator
generate_tsp_file(nb_cities, instance_id,
                  path=tsp_file_path, highest_weight=highest_weight, symmetric=symmetric_instance)
