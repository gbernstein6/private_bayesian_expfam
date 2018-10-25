from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from distributions import Beta_Binomial, Dirichlet_Multinomial
from Gibbs import calculate_Gibbs_posterior


def main():

    conjugate_pair = Beta_Binomial()
    # conjugate_pair = Dirichlet_Multinomial()

    # number of individuals
    N = 1e3

    # privacy settings
    epsilon = .1

    model_parameters = conjugate_pair.draw_model_parameters(conjugate_pair.prior_parameters)

    sufficient_statistics = conjugate_pair.draw_sufficient_statistics(N, model_parameters)

    noise = draw_laplace_noise(conjugate_pair, epsilon)

    noisy_sufficient_statistics = sufficient_statistics + noise

    Gibbs_posterior = calculate_Gibbs_posterior(noisy_sufficient_statistics, conjugate_pair, N, epsilon)

    non_private_posterior = calculate_conjugate_update_posterior(sufficient_statistics, Gibbs_posterior.shape[1], conjugate_pair, N)

    naive_posterior = calculate_conjugate_update_posterior(noisy_sufficient_statistics, Gibbs_posterior.shape[1], conjugate_pair, N)

    plot_posterior(non_private_posterior, naive_posterior, Gibbs_posterior, model_parameters)


def draw_laplace_noise(conjugate_pair, epsilon):

    laplace_scale = 1.0 * conjugate_pair.sensitivity / epsilon

    laplace_noise = np.random.laplace(scale=laplace_scale)

    return laplace_noise


def calculate_conjugate_update_posterior(sufficient_statistics, num_samples, conjugate_pair, N):

    posterior_parameters = conjugate_pair.conjugate_update(N, conjugate_pair.prior_parameters, sufficient_statistics)

    posterior_samples = conjugate_pair.draw_model_parameters(posterior_parameters, size=num_samples)

    return posterior_samples


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
