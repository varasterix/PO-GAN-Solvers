import matplotlib.pyplot as plt
from src.objects import candidateTSP as cTSP


def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + '\n'


def plot_tsp_candidates(candidates):
    cartesian_coordinates_ref = None
    ordered_paths = []
    for i, candidate in enumerate(candidates):
        if not isinstance(candidate, cTSP.CandidateTSP):
            raise Exception('One of the given objects in the list is not an instance of the CandidateTSP class')
        else:
            cartesian_coordinates_ref = candidate.get_cartesian_coordinates() if (i == 0) else cartesian_coordinates_ref
            if (cartesian_coordinates_ref == candidate.get_cartesian_coordinates()).all():
                ordered_paths.append(candidate.to_ordered_path())
            else:
                raise Exception('The given list of candidate TSP has to have the same cartesian coordinates')
    # Plot figure
    if cartesian_coordinates_ref is None:
        raise Exception('There are no cartesian coordinates for this object')
    for oP in ordered_paths:
        if not oP.is_valid_structure():
            raise Exception('One of the given candidates has not a valid structure')
    annotation_gap = 10
    # plt.figure("TSP candidate figure")
    fig, axs = plt.subplots(1, len(ordered_paths))
    plt.suptitle("TSP candidates - Representation of the cycle")
    for i, oP in enumerate(ordered_paths):
        for j, (x, y) in enumerate(ordered_paths[i].get_cartesian_coordinates()):
            axs[i].plot(x, y, "ok")
            axs[i].annotate(j, (x + annotation_gap, y + annotation_gap))
        label = "Not a TSP solution" if not oP.is_solution() else "Solution, D=" + str(oP.distance())
        x_seq, y_seq = [], []
        for city in oP.get_candidate():
            x_seq.append(oP.get_cartesian_coordinates()[city, 0])
            y_seq.append(oP.get_cartesian_coordinates()[city, 1])
        x_seq.append(oP.get_cartesian_coordinates()[oP.get_candidate()[0], 0])
        y_seq.append(oP.get_cartesian_coordinates()[oP.get_candidate()[0], 1])
        axs[i].plot(x_seq, y_seq, label=label)
        axs[i].legend()
    plt.show()
