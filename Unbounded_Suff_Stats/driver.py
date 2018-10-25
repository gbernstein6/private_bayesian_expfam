from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from util import generate_noisy_sufficient_statistics
from util import calculate_conjugate_update_posterior
from Gibbs import calculate_Gibbs_posterior
from distributions import Gamma_Exponential


# prior-model distributions (define as global so autograd can access)
conjugate_pair = Gamma_Exponential()


def main():

    # number of individuals
    N = 1e3

    # privacy settings
    epsilon = .1

    model_parameters = conjugate_pair.draw_model_parameters(conjugate_pair.prior_parameters)

    sufficient_statistics, noisy_sufficient_statistics, noisy_inside_sufficient_statistics, truncation_bounds \
        = generate_noisy_sufficient_statistics(N, epsilon, conjugate_pair, model_parameters)

    Gibbs_posterior = calculate_Gibbs_posterior(noisy_inside_sufficient_statistics, N, epsilon, truncation_bounds)
    non_private_posterior = calculate_conjugate_update_posterior(sufficient_statistics, Gibbs_posterior.shape[1], conjugate_pair, N)
    naive_posterior = calculate_conjugate_update_posterior(noisy_sufficient_statistics, Gibbs_posterior.shape[1], conjugate_pair, N)

    plot_posterior(non_private_posterior, naive_posterior, Gibbs_posterior, model_parameters)


def plot_posterior(non_private_posterior, naive_posterior, Gibbs_posterior, model_parameters):

    # only try to plot a single parameter in case there are multiple
    parameter_index = 0

    Gibbs_parameter_posterior = Gibbs_posterior[parameter_index, :]
    non_private_parameter_posterior = non_private_posterior[parameter_index, :]
    naive_parameter_posterior = naive_posterior[parameter_index, :]

    min_bin = min((min(Gibbs_parameter_posterior), min(non_private_parameter_posterior), min(naive_parameter_posterior)))
    max_bin = max((max(Gibbs_parameter_posterior), max(non_private_parameter_posterior), max(naive_parameter_posterior)))
    bins = np.linspace(min_bin, max_bin, 50)

    counts, _, _ = plt.hist(non_private_parameter_posterior, bins=bins, alpha=0.5, color='gray', label='Non-Private')

    plt.hist(naive_parameter_posterior, bins=bins, alpha=0.5, color='r', label='Naive')
    plt.hist(Gibbs_parameter_posterior, bins=bins, alpha=0.5, color='b', label='Gibbs')

    plt.plot([model_parameters[parameter_index], model_parameters[parameter_index]], [0, max(counts)],
             'k--', label='True Parameter'
             )

    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
