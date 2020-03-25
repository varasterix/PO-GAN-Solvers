import numpy as np
import random
import src.objects.orderedPathBinaryMatrix as oPBM


def kronecker(a, b):
    return 1 if (a == b) else 0


def are_penalty_parameters_correct(params):
    return type(params) == dict and 'A' in params and 'B' in params and 'C' in params and 'D' in params


class HopfieldTSPNetwork:
    def __init__(self, nb_cities, distance_matrix, cartesian_coordinates=None, penalty_parameters=None):
        """
        Initializes the Hopfield TSP Network with a given instance of the Travelling Salesperson Problem
        :param nb_cities: number of cities of the considered TSP instance
        :param distance_matrix: distance/cost/weight matrix of the considered TSP instance
        :param cartesian_coordinates: cartesian coordinates of the considered TSP instance
        :param penalty_parameters: penalty parameters to run the Hopfield TSP Network
        """
        if not are_penalty_parameters_correct(penalty_parameters):
            penalty_parameters = {'A': 500, 'B': 500, 'C': 200, 'D': 500}
        self.nb_cities = nb_cities  # n
        self.distance_matrix = distance_matrix
        self.cartesian_coordinates = cartesian_coordinates
        self.nb_neurons = nb_cities * nb_cities  # n²
        self.params = penalty_parameters  # PENALTY PARAMETERS
        self.connection_weights = self.__compute_connection_weights()  # W_i,j,k,l (n²,n²) - PARAMETERS
        self.external_inputs = np.array([self.params['C'] * nb_cities] * self.nb_neurons)  # I_i,j (n²,1) - PARAMETERS
        self.constant_energy_term = (self.params['C'] * nb_cities * nb_cities) / 2  # PARAMETER
        self.internal_states = np.zeros(self.nb_neurons)  # u_i,j (n²,1) - VARIABLES
        self.external_states = np.array([random.randint(0, 1)] * self.nb_neurons)  # v_i,j (n²,1) - VARIABLES -> 0 ?
        self.activation_function = lambda d: 1 if (d > 0) else 0  # f(), discrete version
        self.best_configuration = self.external_states.copy()

    def get_nb_cities(self):
        return self.nb_cities

    def get_nb_neurons(self):
        return self.nb_neurons

    def get_distance_matrix(self):
        return self.distance_matrix

    def get_distance_from_to(self, i, j):
        """
        Returns the distance from the city i to  the city j
        :param i: city i
        :param j: city j
        :return: the distance from the city i to the city j
        """
        return self.distance_matrix[i][j]

    def get_cartesian_coordinates(self):
        return self.cartesian_coordinates

    def __compute_connection_weights(self):
        weights = np.zeros((self.get_nb_neurons(), self.get_nb_neurons()))
        for i in range(self.get_nb_cities()):
            for j in range(self.get_nb_cities()):
                for k in range(self.get_nb_cities()):
                    for l in range(self.get_nb_cities()):
                        from_neuron = (i * self.get_nb_cities()) + j
                        to_neuron = (k * self.get_nb_cities()) + l
                        weights[from_neuron][to_neuron] = \
                            (-1) * self.params['A'] * kronecker(i, k) * (1 - kronecker(j, l)) \
                            - self.params['B'] * kronecker(j, l) * (1 - kronecker(i, k)) - self.params['C'] \
                            - self.params['D'] * self.get_distance_from_to(i, k) * \
                            (kronecker(l, j - 1) + kronecker(l, j + 1))
        return weights

    def get_weight(self, i, j, k, l):
        from_neuron = (i * self.get_nb_cities()) + j
        to_neuron = (k * self.get_nb_cities()) + l
        if from_neuron >= len(self.connection_weights) or to_neuron >= len(self.connection_weights):
            raise IndexError("Out of range: from_neuron: {}, to_neuron: {}".format(from_neuron, to_neuron))
        return self.connection_weights[from_neuron][to_neuron]

    def add_weight(self, i, j, k, l, value):
        from_neuron = (i * self.get_nb_cities()) + j
        to_neuron = (k * self.get_nb_cities()) + l
        if from_neuron >= len(self.connection_weights) or to_neuron >= len(self.connection_weights):
            raise IndexError("Out of range: from_neuron: {}, to_neuron: {}".format(from_neuron, to_neuron))
        self.connection_weights[from_neuron][to_neuron] += value

    def set_weight(self, i, j, k, l, value):
        from_neuron = (i * self.get_nb_cities()) + j
        to_neuron = (k * self.get_nb_cities()) + l
        if from_neuron >= len(self.connection_weights) or to_neuron >= len(self.connection_weights):
            raise IndexError("Out of range: from_neuron: {}, to_neuron: {}".format(from_neuron, to_neuron))
        self.connection_weights[from_neuron][to_neuron] = value

    def get_external_input(self, i, j):
        index = (i * self.get_nb_cities()) + j
        return self.external_inputs[index]

    def get_best_configuration(self, i, j):
        index = (i * self.get_nb_cities()) + j
        return self.best_configuration[index]

    def get_external_state(self, i, j):
        index = (i * self.get_nb_cities()) + j
        return self.external_states[index]

    def get_internal_state(self, i, j):
        index = (i * self.get_nb_cities()) + j
        return self.internal_states[index]

    def compute_external_state(self, i, j):
        index = (i * self.get_nb_cities()) + j
        self.external_states[index] = self.activation_function(self.get_internal_state(i, j))

    def compute_internal_state(self, i, j):
        index = (i * self.get_nb_cities()) + j
        stimulation = 0
        for k in range(self.get_nb_cities()):
            for l in range(self.get_nb_cities()):
                stimulation += self.get_weight(i, j, k, l) * self.get_external_state(k, l)
        self.internal_states[index] = stimulation + self.get_external_input(i, j)

    def clear(self):
        """
        Clear any connection weights.
        """
        for from_neuron in range(self.get_nb_neurons()):
            for to_neuron in range(self.get_nb_neurons()):
                self.connection_weights[from_neuron][to_neuron] = 0

    def compute_energy(self):
        """
        Compute the TSP energy function associated to this Hopfield TSP Network
        :return: the current energy of the network. The network will seek to minimize this value
        """
        energy_1 = 0  # The quadratic term is computed
        for i in range(self.get_nb_cities()):
            for j in range(self.get_nb_cities()):
                for k in range(self.get_nb_cities()):
                    for l in range(self.get_nb_cities()):
                        energy_1 += self.get_weight(i, j, k, l) * self.get_external_state(i, j) \
                                    * self.get_external_state(k, l)
        energy_2 = 0  # The linear term is computed
        for i in range(self.get_nb_cities()):
            for j in range(self.get_nb_cities()):
                energy_2 += self.get_external_input(i, j) * self.get_external_state(i, j)
        return (-1) * energy_1 / 2 - energy_2 + self.constant_energy_term  # TSP energy function : constant term added

    def reset(self):
        """
        Resets the neural network to random external states, and internal states to zero
        """
        for i in range(self.get_nb_neurons()):
            self.external_states[i] = random.randint(0, 1)
        for i in range(self.get_nb_neurons()):
            self.internal_states[i] = 0

    def get_all_neurons_indices(self):
        neurons_indices = []
        for i in range(self.nb_cities):
            for j in range(self.nb_cities):
                neurons_indices.append((i, j))
        return neurons_indices

    def run(self, neurons_indices):
        """
        Performs one Hopfield iteration (updates the internal and external of the n² neurons of the TSP network)
        :param neurons_indices: all the pairs of indices corresponding to one neuron of the Hopfield TSP Network
        """
        random.shuffle(neurons_indices)
        for i, j in neurons_indices:
            self.compute_internal_state(i, j)
            self.compute_external_state(i, j)

    def run_until_stable(self, max_iterations=500, stop_at_local_min=True):
        """
        Run the network until the energy function has reached a local minimum if stop_at_local_min is True, or until it
        becomes stable and does not change from more runs otherwise
        :param max_iterations: The maximum number of iterations/cycles to run before giving up
        :param stop_at_local_min: boolean to stop running if a local minimum is reached by the energy function
        :return: The number of iterations/cycles that were run, and the TSP energy over these iterations (without
        the random initial TSP energy in the network)
        """
        neurons_indices = self.get_all_neurons_indices()
        stop_criterion = False
        current_external_states = self.external_states.copy()
        lowest_energy = self.compute_energy()
        energy = []
        iteration = 0
        while not stop_criterion:
            iteration += 1
            self.run(neurons_indices)
            new_external_states = self.external_states.copy()
            new_energy = self.compute_energy()
            energy.append(new_energy)
            # The stop criterion is checked
            if stop_at_local_min and len(energy) >= 2 and energy[-2] < new_energy:
                # if stop_at_local_min and lowest_energy == new_energy:
                stop_criterion = True
            elif (new_external_states == current_external_states).all():
                stop_criterion = True
            else:
                if iteration >= max_iterations:
                    stop_criterion = True
            # The best configuration is updated
            if new_energy < lowest_energy:
                self.best_configuration = new_external_states.copy()
                lowest_energy = new_energy
            current_external_states = new_external_states.copy()
        return iteration, energy

    def get_best_ordered_path_binary_matrix(self):
        binary_matrix = np.zeros((self.get_nb_cities(), self.get_nb_cities()), dtype=int)
        for i in range(self.get_nb_cities()):
            for j in range(self.get_nb_cities()):
                binary_matrix[i, j] = self.get_best_configuration(i, j)
        return oPBM.OrderedPathBinaryMatrix(binary_matrix, self.get_distance_matrix(), self.get_cartesian_coordinates())
