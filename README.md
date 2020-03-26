# PO-GAN-Solvers

The initial goal of this project is to create an architecture close to the GAN architecture, where the Discriminator is 
replaced with a solver related to the concerned problem (a constraint satisfaction/optimization problem).

The source code (in Python and Java) currently summarizes some neural network architectures and methods used to solve 
a well-known COP : the Travelling Salesperson Problem.

In the following parts, we present the crucial information to understand the content of this project, which was 
developed by four engineers students, between October 2019 and March 2020. 

## Database

Considering an instance of the TSP, it is defined with its weight/distance matrix *M* of size (*n* x *n*) with *n*, the
number of cities of this considered instance.

Thus, a file in the "data/tsp_files/", called "dataSet_*n*_*instance_id*.tsp" is written in this format :

    <instance_id>   (between 0 and the number of instances with <n> cities)
    <n>             (number of cities)
    <M>[0,0]   <M>[0,1]    ...   <M>[0,n-1]
    <M>[1,0]   <M>[1,1]    ...   <M>[1,n-1]
    ...        ...         ...   ...
    <M>[n-1,0] <M>[n-1,1]  ...   <M>[n-1,n-1]

Notes: 
- For all *i* in [| *0*, *n-1* |], *M* [*i*, *i*] *= 0*
- *M* [*i*, *j*] represents the weight/distance from the city *i* to the city *j* (as an integer between *0* and an 
highest integer weight *h* included)


Besides, a file in the same repository, called "dataSet_*n*_ *instance_id*.heuristic", which also contains a solution 
to the considered instance of the TSP (in the "dataSet_*n*_*instance_id*.tsp" file), is written in this format :
    
    <instance_id>
    <n>
    <M>[0,0]   <M>[0,1]    ...  <M>[0,n-1]
    <M>[1,0]   <M>[1,1]    ...  <M>[1,n-1]
    ...        ...         ...  ...
    <M>[n-1,0] <M>[n-1,1]  ...  <M>[n-1,n-1]
    <S>[0]     <S>[1]      ...  <S>[n-1]
    <total_weigh>

Notes:
- *S* [*i*] represents the (*i+1*) th city visited in the solution
- *total_weight* represents the total cost/weight/distance of the solution *S* for the considered instance

This format is respected in all the data files in the repository data/, just with a variable extension.


## Neural Network Architectures

Five main neural network architectures were developed (using the library PyTorch for most of them) :

- A simple neural network (main code in src/optimization_net/)

- A GAN (Generative Adversarial Network) (src/gan/ - src/generator/ - src/discriminator/)

- A segmented neural network (src/segmented_net/)

- A Hopfield neural network (src/hopfield_net/)

- A Deep Q-Learning architecture for reinforcement learning (src/DQN/) (whose implementation is not finished) 

For these five architectures, the executive/train scripts are in the following respective paths :

- src/optimization_net/structureLearning.py or src/optimization_net/chocoLearning.py

- src/gan/gan.py

- src/segmented_net/segmentedLearning.py

- src/hopfield_net/hopfieldLearning.py

- src/DQN/train.py


## Shell scripts (DEPRECATED)

#### How to use the shell file "generator.sh" :

This shell file is used to generate the TSP dataSet file "dataSet_*n*_*instance_id*.tsp". 

This file corresponds to the *i* th instance of the TSP with *n* cities in the repository at
the path *p* (by default, the file is written in the repository "data/tsp_files/" from the 
project root) with *h* the highest integer weight generated of the instance (by default, it is 100).
The option *s* indicates if the TSP instance has to be symmetric or not (by default, non symmetric).

Command line to use at the root project :

    sh generator.sh -n [nb_cities] -i [instance_id]

Optional arguments of this command :

    -p [path_to_tsp_dataSet_file]
    -h [highest_integer_weight]
    -s [is_weight_matrix_symmetric]

Note: 
- the option -p need to accept files with absolute repositories (like the others -p options after).
- the option -s need to accept only the integers 0 (for boolean False) and (for boolean True)

Example of TSP file "dataSet_*10*_*0*.tsp" in "test/" repository with *h = 5* (non symmetric):

    sh generator.py -n 10 -i 0 -h 5 -p "test/" -s 0

#### How to use the shell file "multi_generator.sh" :

This shell file is used to generate *nb_instances* TSP dataSet files "dataSet_*n*_*i*.tsp" (with *i* between *f* and 
(*f + nb_instances - 1*)). 

This files are stored in the repository at the path *p* (by default, the file is written in the repository 
"data/tsp_files/" from the project root) with *h* the highest integer weight generated for the instances (by default, 
it is *100*). *s* indicates if the TSP instances have to be symmetric or not (by default, non symmetric). *f* indicates 
the first instance id to generate (by default, it is *0*).

Command line to use at the root project :

    sh multi_generator.sh -n [nb_cities] -x [nb_instances]

Optional arguments of this command :

    -p [path_to_tsp_dataSet_files]
    -h [highest_integer_weight]
    -s [is_weight_matrices_symmetric]
    -f [first_instance_id]


#### How to use the shell file "compute_heuristic.sh" :

###### WARNING: From here, TSP instances with symmetric weight/distance/cost matrix are considered

This shell file is used to generate the TSP dataSet heuristic solution file "dataSet_*n*_*instance_id*.heuristic". 

This file corresponds to the *i* th instance of the TSP with *n* cities which is given by the TSP file 
"dataSet_*n*_*instance_id*.tsp" in the repository at the path *p* (by default, "data/tsp_files/" from the 
project root). This heuristic solution file is stored at the same path *p*. The heuristic solution is computed 
with the nearest neighbor greedy heuristic and then improved with the 2-opt heuristic during *l* seconds (10s by default).

Command line to use at the root project :

    sh compute_heuristic.sh -n [nb_cities] -i [instance_id]

Optional arguments of this command :

    -p [path_to_tsp_dataSet_file]
    -l [time_limit_two_opt]

#### How to use the shell file "multi_compute_heuristic.sh" :

This shell file is used to generate *nb_instances* TSP dataSet heuristic solution files "dataSet_*n*_*i*.heuristic" 
(with *i* between *f* and (*f + nb_instances - 1*)). 

This files are stored in the repository at the path *p* (by default, "data/tsp_files/") with *l* the time in seconds to 
improve the nearest neighbor greedy solution with 2-opt heuristic improvement (10s by default).
*f* indicates the first instance id to solve (by default, it is *0*).

Command line to use at the root project :

    sh multi_compute_heuristic.sh -n [nb_cities] -x [nb_instances]

Optional arguments of this command :

    -p [path_to_tsp_dataSet_files]
    -l [time_limit_two_opt]
    -f [first_instance_id]
