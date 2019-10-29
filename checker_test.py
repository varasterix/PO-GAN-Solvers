import checker as c

nb_pb = 0

if c.nb_cycles([1, 2, 3, 0]) != 1:
    nb_pb += 1
    print("Problem: the proper solution [1, 2, 3, 0] was detected as a improper")

if c.nb_cycles([0, 1, 2]) == 1:
    nb_pb += 1
    print("Problem: the improper solution [0, 1, 2] was detected as proper")

if c.nb_cycles([0, 1, 2, 1]):
    nb_pb += 1
    print("Problem: the improper solution [0, 1, 2, 1] was detected as proper")

if nb_pb == 0:
    print("No Problem was detected by the unit tests")
