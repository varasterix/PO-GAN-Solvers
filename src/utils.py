import math
import matplotlib.pyplot as plt
from src.objects import candidateTSP as cTSP


def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + '\n'


def plot_tsp_candidates(candidates, instance_id, label_titles=None, save_path=None, show_figure=False):
    # Exceptions control
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
    nb_ordered_paths = len(ordered_paths)
    sqrt_ordered_paths = math.ceil(math.sqrt(nb_ordered_paths))
    annotation_gap = 10
    fig, axs = plt.subplots(nb_ordered_paths // sqrt_ordered_paths, sqrt_ordered_paths)
    plt.suptitle("TSP candidates instance {} - Representation of the cycles".format(instance_id))
    for i, oP in enumerate(ordered_paths):
        ax = axs[i] if (nb_ordered_paths // sqrt_ordered_paths <= 1) \
            else axs[i // sqrt_ordered_paths, i % sqrt_ordered_paths]
        for j, (x, y) in enumerate(oP.get_cartesian_coordinates()):
            ax.plot(x, y, 'ok')
            ax.annotate(j, (x + annotation_gap, y + annotation_gap))
        label = "Not TSP solution" if not oP.is_solution() else "D=" + str(oP.distance())
        x_seq, y_seq = [], []
        for city in oP.get_candidate():
            x_seq.append(oP.get_cartesian_coordinates()[city, 0])
            y_seq.append(oP.get_cartesian_coordinates()[city, 1])
        x_seq.append(oP.get_cartesian_coordinates()[oP.get_candidate()[0], 0])
        y_seq.append(oP.get_cartesian_coordinates()[oP.get_candidate()[0], 1])
        ax.plot(x_seq, y_seq, label=label)
        ax.set_title('' if label_titles is None else label_titles[i])
        ax.legend()
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    if save_path is not None:
        plt.savefig(save_path)
    if show_figure:
        plt.show()
