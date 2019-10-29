

# Counts the number of cycles in the solution. solution[i] == i is counted as a cycle.
# parameter: solution: a solution to the TSP
# return: the number of cycles in the solution
def nb_cycles(solution):
    visited = [0] * len(solution)
    nb = 0
    for i in range(len(solution)):
        j = i
        if visited[j] == 0:
            nb += 1
        while visited[j] < 1:
            visited[j] += 1
            j = solution[j]
    return nb
