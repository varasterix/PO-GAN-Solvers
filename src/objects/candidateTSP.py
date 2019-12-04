from src.objects.candidateCSP import CandidateCSP


class CandidateTSP(CandidateCSP):  # TODO : abus de langage "distance matrix" -> "weight matrix"
    """
    Abstract class specifying an interface of a candidate of an instance of the Travelling Salesman Problem (TSP)
    """

    def is_solution(self):
        raise NotImplementedError

    def distance(self):
        if not self.is_solution():
            raise Exception('The candidate is not a solution of the TSP')
        else:
            raise NotImplementedError
