"""Do stats related to the ATNx cell counts."""

from scipy.special import comb

from mpmath import quad


def C(n, r):
    """Return n choose r."""
    return comb(n, r)


def binomial(n, r, p):
    """Return prob of r successes from n draws with probability p of success."""
    return C(n, r) * (p ** r) * ((1 - p) ** (n - r))


class binomial_bayes(object):
    """
    A bayesian statistics calculation based on comparing
    two sets of samples which are drawn from two different
    binomial distributions.
    
    Note 1 indicates control, while 2 indicates non-control.

    Attributes
    ----------
    a1 : float
        The total samples drawn from control population.
    a2 : float
        The total successes drawn from control population.
    b1 : float
        The total samples drawn from non-control population.
    b2 : float
        The total successes drawn from non-control population.
    prior_fn : function
        The function representing the prior distribution.
    pe : float
        The probability of the evidence.

    """

    def __init__(self, a1, a2, b1, b2, prior_fn):
        """
        Initialise the stats object

        Parameters
        ----------
        a1 : float
            The total samples drawn from control population.
        a2 : float
            The total successes drawn from control population.
        b1 : float
            The total samples drawn from non-control population.
        b2 : float
            The total successes drawn from non-control population.
        prior_fn : function
            The function representing the prior distribution.
        
        """
        self.a1 = a1
        self.a2 = a2
        self.b1 = b1
        self.b2 = b2
        self.prior_fn = prior_fn
        self.pe = self.prob_evidence()

    def likelihood(self, p1, p2):
        """Likelihood of p1 being prob success control, p2 prob success non."""
        return binomial(self.a1, self.b1, p1) * binomial(self.a2, self.b2, p2)

    def prior(self, p1, p2):
        """Prior of p1 being prob success control, p2 prob success non."""
        return self.prior_fn(p1, p2)

    def integrand(self, p1, p2):
        """Prior times likelihood."""
        return self.likelihood(p1, p2) * self.prior(p1, p2)

    def prob_evidence(self):
        """Prior times likelihood integrated over square [0,1], [0,1]."""
        return self.do_integration(0, 1, 0, 1)

    def do_integration(self, lower1, upper1, lower2, upper2):
        """Integrate the integrand over the given rectangle [l1, u1], [l2, u2]."""
        return quad(self.integrand, [lower1, upper1], [lower2, upper2])


def uniform_prior(p1, p2):
    """Return 1."""
    return 1


def main(
    num_ctrl_records,
    num_ctrl_success,
    num_lesion_records,
    num_lesion_success,
    bayes_ctrl_prob=0.2,
):
    bb = binomial_bayes(
        num_ctrl_records,
        num_lesion_records,
        num_ctrl_spatial_records,
        num_lesion_spatial_records,
        uniform_prior,
    )
    bb_result = bb.do_integration(0, bayes_ctrl_prob, 0, 1.0) / bb.pe


if __name__ == "__main__":
    num_ctrl_records = 44
    num_ctrl_spatial_records = 4
    num_lesion_records = 37
    num_lesion_spatial_records = 0

    res = main(
        num_ctrl_records,
        num_ctrl_spatial_records,
        num_lesion_records,
        num_lesion_spatial_records,
        bayes_ctrl_prob=0.2,
    )
    print(res)
