from src.objects.candidateCSP import CandidateCSP


class CandidateTSP(CandidateCSP):
    """
    Abstract class specifying an interface of a candidate of an instance of the Travelling Salesman Problem (TSP)
    """

    def __eq__(self, other):
        raise NotImplementedError

    def __ge__(self, other):  # equivalent to operator >=
        if not isinstance(other, CandidateTSP):  # don't attempt to compare against unrelated types
            return NotImplemented
        else:
            if not self.is_solution():
                raise Exception('The comparisons functions only compare two solutions of a same instance of the TSP')
            if not other.is_solution():
                raise Exception('The comparisons functions only compare two solutions of a same instance of the TSP')
            if self.get_nb_cities() != other.get_nb_cities() \
                    or not (self.get_weight_matrix() == other.get_weight_matrix()).all():
                raise Exception('The comparisons functions only compare two solutions of a same instance of the TSP')
            else:
                return self.distance() >= other.distance()

    def __gt__(self, other):  # equivalent to operator >
        if not isinstance(other, CandidateTSP):  # don't attempt to compare against unrelated types
            return NotImplemented
        else:
            if not self.is_solution():
                raise Exception('The comparisons functions only compare two solutions of a same instance of the TSP')
            if not other.is_solution():
                raise Exception('The comparisons functions only compare two solutions of a same instance of the TSP')
            if self.get_nb_cities() != other.get_nb_cities() \
                    or not (self.get_weight_matrix() == other.get_weight_matrix()).all():
                raise Exception('The comparisons functions only compare two solutions of a same instance of the TSP')
            else:
                return self.distance() > other.distance()

    def __lt__(self, other):  # equivalent to operator <
        if not isinstance(other, CandidateTSP):  # don't attempt to compare against unrelated types
            return NotImplemented
        else:
            if not self.is_solution():
                raise Exception('The comparisons functions only compare two solutions of a same instance of the TSP')
            if not other.is_solution():
                raise Exception('The comparisons functions only compare two solutions of a same instance of the TSP')
            if self.get_nb_cities() != other.get_nb_cities()\
                    or not (self.get_weight_matrix() == other.get_weight_matrix()).all():
                raise Exception('The comparisons functions only compare two solutions of a same instance of the TSP')
            else:
                return self.distance() < other.distance()

    def __le__(self, other):  # equivalent to operator <=
        if not isinstance(other, CandidateTSP):  # don't attempt to compare against unrelated types
            return NotImplemented
        else:
            if not self.is_solution():
                raise Exception('The comparisons functions only compare two solutions of a same instance of the TSP')
            if not other.is_solution():
                raise Exception('The comparisons functions only compare two solutions of a same instance of the TSP')
            if self.get_nb_cities() != other.get_nb_cities() \
                    or not (self.get_weight_matrix() == other.get_weight_matrix()).all():
                raise Exception('The comparisons functions only compare two solutions of a same instance of the TSP')
            else:
                return self.distance() <= other.distance()

    def is_solution(self):
        raise NotImplementedError

    def distance(self):
        if not self.is_solution():
            raise Exception('The candidate is not a solution of the TSP')
        else:
            raise NotImplementedError

    def get_nb_cities(self):
        raise NotImplementedError

    def get_candidate(self):
        raise NotImplementedError

    def get_weight_matrix(self):
        raise NotImplementedError

    def get_cartesian_coordinates(self):
        raise NotImplementedError

    def to_ordered_path(self):
        raise NotImplementedError

    def to_ordered_path_binary_matrix(self):
        raise NotImplementedError

    def to_neighbours(self):
        raise NotImplementedError

    def to_neighbours_binary_matrix(self):
        raise NotImplementedError
