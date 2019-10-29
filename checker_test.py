import checker as c

nb_pb = 0

if c.nb_cycles([1, 2, 3, 0]) != 1:
    nb_pb += 1
    print("Problème: le solution simple sans cycle [1, 2, 3, 0] a été détectée comme une mauvaise solution")

if c.nb_cycles([0, 1, 2]) == 1:
    nb_pb += 1
    print("Problème: la mauvaise solution [0, 1, 2] a été détectée comme une bonne solution")

if c.nb_cycles([0, 1, 2, 1]):
    nb_pb += 1
    print("Problème: la mauvaise solution [0, 1, 2, 1] a été détectée comme une bonne solution")

if nb_pb == 0:
    print("Aucun problème détecté par les tests unitaires")
