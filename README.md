# PO-GAN-Solvers

The goal of this project is to create an architecture close to the GAN architecture, where the Discriminator is replaced with a solver related to the concerned problem.

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

## Shell scripts

#### How to use the shell file "generator.sh" :

This shell file is used to generate the TSP dataSet file "dataSet_*n*_*instance_id*.tsp". 

This file corresponds to the *i* th instance of the TSP with *n* cities in the repository at
the path *p* (by default, the file is written in the repository "data/tsp_files/" from the 
project root) with *h* the highest integer weight generated of the instance (by default, it is 100).

Command line to use at the root project :

    sh generator.sh -n [nb_cities] -i [instance_id]

Optional arguments of this command :

    -p [path_to_tsp_dataSet_file]
    -h [highest_integer_weight]

Note: the option -p need to accept files with absolute repositories (like the others -p options after).

Example of TSP file "dataSet_*10*_*0*.tsp" in "test/" repository with *h = 5* :

    sh generator.py -n 10 -i 0 -h 5 -p "test/"

#### How to use the shell file "multi_generator.sh" :

This shell file is used to generate *nb_instances* TSP dataSet files "dataSet_*n*_*i*.tsp" (with *i* between *0* and 
*(nb_instances-1)*). 

This files are stored in the repository at the path *p* (by default, the file is written in the repository 
"data/tsp_files/" from the project root) with *h* the highest integer weight generated for the instances (by default, 
it is 100).

Command line to use at the root project :

    sh multi_generator.sh -n [nb_cities] -x [nb_instances]

Optional arguments of this command :

    -p [path_to_tsp_dataSet_files]
    -h [highest_integer_weight]

#### How to use the shell file "compute_heuristic.sh" :

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
(with *i* between *0* and *(nb_instances-1)*). 

This files are stored in the repository at the path *p* (by default, "data/tsp_files/") with *l* the time in seconds to 
improve the nearest neighbor greedy solution with 2-opt heuristic improvement (10s by default).

Command line to use at the root project :

    sh multi_compute_heuristic.sh -n [nb_cities] -x [nb_instances]

Optional arguments of this command :

    -p [path_to_tsp_dataSet_files]
    -l [time_limit_two_opt]
