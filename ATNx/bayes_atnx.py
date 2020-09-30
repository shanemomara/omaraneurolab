"""Do stats related to the ATNx cell counts."""

import os

from scipy.special import comb
from scipy.stats import chisquare
from scipy.stats import chi2_contingency
from scipy.stats import fisher_exact
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
import numpy as np

from skm_pyutils.py_config import read_python


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

    def __init__(self, a1, b1, a2, b2, prior_fn):
        """
        Initialise the stats object

        Parameters
        ----------
        a1 : float
            The total samples drawn from control population.
        b1 : float
            The total successes drawn from control population.
        a1 : float
            The total samples drawn from non-control population.
        b2 : float
            The total successes drawn from non-control population.
        prior_fn : function
            The function representing the prior distribution.
        
        """
        self.a1 = a1
        self.b1 = b1
        self.a2 = a2
        self.b2 = b2
        self.prior_fn = prior_fn
        self.pe = self.prob_evidence()

    def likelihood(self, p1, p2):
        """Likelihood of p1 being prob success control, p2 prob success non."""
        return binomial(self.a1, self.b1, p1) * binomial(self.a2, self.b2, p2)

    def prior(self, p1, p2):
        """Prior of p1 being prob success control, p2 prob success non."""
        return self.prior_fn(p1, p2)

    def integrand(self, p2, p1):
        """Prior times likelihood."""
        return self.likelihood(p1, p2) * self.prior(p1, p2)

    def prob_evidence(self):
        """Prior times likelihood integrated over square [0,1], [0,1]."""
        return self.do_integration(0, 1, 0, 1)

    def do_integration(self, lower1, upper1, lower2, upper2):
        """Integrate the integrand over the given rectangle [l1, u1], [l2, u2]."""
        return dblquad(
            self.integrand, lower1, upper1, lambda x: lower2, lambda x: upper2
        )[0]

    def do_integration_tri(self, lower1, upper1, percent):
        """Integrate the integrand over the given triangle [l1, 0], [u1,0], [u1, percent]."""
        return dblquad(
            self.integrand, lower1, upper1, lambda x: 0, lambda x: percent * x
        )[0]

    def plot_posterior(self, srate=100):
        samples = np.linspace(0, 1, srate)
        res = np.zeros(shape=(srate,))
        fig, ax = plt.subplots()

        sum_val = 0
        for i, val in enumerate(samples):
            bin_res = binomial(self.a1, self.b1, val)
            sum_val += bin_res
            res[i] = bin_res
        ax.plot(samples, res, label="control", c="k")
        sum_val = 0
        for i, val in enumerate(samples):
            bin_res = binomial(self.a2, self.b2, val)
            sum_val += bin_res
            res[i] = bin_res
        ax.plot(samples, res, label="lesion", c="k", linestyle="--")
        ax.set_xlabel("Probability of spatial recording")
        ax.set_ylabel("Posterior probability")
        plt.legend()

        fig.savefig("2d.png", dpi=400)

    def plot_integrand(self, srate=100, percent=1.0):
        samples = np.linspace(0, 1, srate)
        samples_x = np.tile(samples, srate)
        samples_y = np.repeat(samples, srate)
        samples_z = np.zeros(shape=(srate * srate))

        for i, (x, y) in enumerate(zip(samples_x, samples_y)):
            res = self.integrand(y, x)
            samples_z[i] = res

        fig = plt.figure()
        ax = fig.gca(projection="3d")
        surf = ax.plot_trisurf(
            samples_x, samples_y, samples_z, cmap="viridis", linewidth=0.2
        )
        fig.colorbar(surf, shrink=0.5, aspect=5)
        # ax.view_init(30, 45)

        ax.set_xlabel("Probability of control spatial")
        ax.set_ylabel("Probability of lesion spatial")
        ax.set_zlabel("Joint posterior probability")

        fig.savefig("3d.png", dpi=400)

        fig, ax = plt.subplots()
        samples_z = np.zeros(shape=(srate, srate))
        for i, x in enumerate(samples):
            for j, y in enumerate(samples):
                res = self.integrand(y, x)
                samples_z[j, i] = res
        surf = ax.contour(
            samples, samples, samples_z.reshape(srate, srate), cmap="viridis"
        )
        fig.colorbar(surf, shrink=0.5, aspect=5)

        ax.plot(samples, samples * percent, c="k", linestyle="--")

        ax.set_xlabel("Probability of control")
        ax.set_ylabel("Probability of lesion")

        fig.savefig("contour.png", dpi=400)

    def __str__(self):
        """Return this object as a string."""
        return "CTRL {}, SUCC {}, LES {}, SUCC {}".format(
            self.a1, self.a2, self.b1, self.b2
        )


def uniform_prior(p1, p2):
    """Return 1."""
    return 1


def parse_numbers(file_location):
    """Parse the data stored in python."""
    # (animal, date, num_spatial, num_non, include)
    data = read_python(file_location)
    control = data.get("control", [])
    lesion = data.get("lesion", [])

    # num_records, num_spatial_records, num_non_spatial_records, total_spatial, total_non_spatial
    arr = np.zeros(shape=(2, 5), dtype=np.int32)

    for i, arr_data in enumerate([control, lesion]):
        for val in arr_data:
            if val[-1]:
                arr[i, 0] += 1
                arr[i, 3] += val[2]
                arr[i, 4] += val[3]
                if val[2] > 0:
                    arr[i, 1] += 1
                if val[3] > 0:
                    arr[i, 2] += 1

    return data, arr


def get_contingency(data):
    f_obs = np.zeros(shape=(2, 2), dtype=np.int32)
    f_obs[0, 0] = data[0, 1]
    f_obs[1, 0] = data[0, 0] - data[0, 1]
    f_obs[0, 1] = data[1, 1]
    f_obs[1, 1] = data[1, 0] - data[1, 1]
    f_obs_spat = f_obs

    f_obs = np.zeros(shape=(2, 2), dtype=np.int32)
    f_obs[0, 0] = data[0, 2]
    f_obs[1, 0] = data[0, 0] - data[0, 2]
    f_obs[0, 1] = data[1, 2]
    f_obs[1, 1] = data[1, 0] - data[1, 2]
    f_obs_ns = f_obs

    return f_obs_spat, f_obs_ns


def chi_squared(data):
    """
    This was used to check chi2_contingency manually.
    
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html.
    
    """
    f_obs = data

    # Contingency table stats
    A, B = np.sum(data, axis=0)
    C, D = np.sum(data, axis=1)
    T = np.sum(f_obs.flatten())

    # The data comes from a ravelled [2, 2] contingency table
    dof_change = 1

    f_exp = np.zeros(shape=data.shape, dtype=np.float32)
    f_exp[0, 0] = A * C / T
    f_exp[0, 1] = B * C / T
    f_exp[1, 0] = A * D / T
    f_exp[1, 1] = B * D / T

    result = chisquare(f_obs, f_exp, ddof=dof_change, axis=None)

    return result, f_obs, f_exp


def bayes_stats(
    num_ctrl_records,
    num_ctrl_success,
    num_lesion_records,
    num_lesion_success,
    bayes_les_prob=0.2,
    srate=30,
):
    bb = binomial_bayes(
        num_ctrl_records,
        num_ctrl_success,
        num_lesion_records,
        num_lesion_success,
        uniform_prior,
    )
    bb_result = bb.do_integration_tri(0, 1.0, bayes_les_prob) / bb.pe
    bb.plot_integrand(srate, bayes_les_prob)
    bb.plot_posterior(srate)

    return bb_result


def prob_ns(total_samples, s_prob, spat_samples):
    a1 = (1 - s_prob) ** total_samples
    a2 = binomial(total_samples, spat_samples, s_prob)
    return (a1, a2)


def main():
    here = os.path.abspath(os.path.dirname(__file__))
    data_loc = os.path.join(here, "cell_stats.py")
    data, arr = parse_numbers(data_loc)
    f_obs = get_contingency(arr)
    # Note Barnard's exact test from R at
    # https://cran.r-project.org/web/packages/Barnard/
    # Gives similar results (p value wise)
    chi_result_spat = fisher_exact(f_obs[0])
    chi_result_ns = fisher_exact(f_obs[1])
    result_prob = prob_ns(arr[1, 0], arr[0, 1] / arr[0, 0], arr[0, 1])

    num_ctrl_records = arr[0, 0]
    num_ctrl_spatial_records = arr[0, 1]
    num_lesion_records = arr[1, 0]
    num_lesion_spatial_records = arr[1, 1]

    res = bayes_stats(
        num_ctrl_records,
        num_ctrl_spatial_records,
        num_lesion_records,
        num_lesion_spatial_records,
        bayes_les_prob=0.2,
        srate=100,
    )

    result_dict = {}
    result_dict["chi_spat"] = chi_result_spat
    result_dict["chi_ns"] = chi_result_ns
    result_dict["bayes"] = res
    result_dict["prob"] = result_prob

    return result_dict, f_obs[0], f_obs[1]


if __name__ == "__main__":
    result = main()
    print(result[0])
    print(result[1])
    print(result[2])
