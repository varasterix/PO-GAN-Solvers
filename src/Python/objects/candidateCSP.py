class CandidateCSP:
    """
    Abstract class specifying an interface of a candidate of a Constraint Satisfaction Problem (CSP)
    Note: An evaluation of a candidate is consistent if it does not violate any of the constraints.
    An evaluation of a candidate is complete if it includes all variables.
    An evaluation is a candidate if it is consistent and complete; such an evaluation is said to solve the CSP.
    """

    def is_solution(self):
        """Returns a boolean true if the candidate is a solution of the CSP, false otherwise"""
        raise NotImplementedError
