def nb_cycles(solution):
    visitee = [0] * len(solution)
    nb = 0
    for i in range(len(solution)):
        j = i
        if visitee[j] == 0:
            nb += 1
        while visitee[j] < 1:
            visitee[j] += 1
            j = solution[j]
    return nb
