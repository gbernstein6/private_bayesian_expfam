import numpy as np
import scipy, scipy.stats

from util import product_of_two_multivariate_normals
from util import calculate_sensitivity
from parameter_autograd import calculate_sufficient_statistics_section_parameters


def calculate_Gibbs_posterior(noisy_inside_sufficient_statistics, conjugate_pair, N, epsilon, truncation_bounds):
    # returns posterior as numpy array of shape (p, i), where p is number of model parameters and i is number of Gibbs samples

    burnin, num_iterations = 500, 1000

    sensitivity = calculate_sensitivity(conjugate_pair, truncation_bounds)

    model_parameters, noise_covariance = initialize_Gibbs(conjugate_pair, epsilon, sensitivity)

    Gibbs_posterior = np.zeros((len(model_parameters), num_iterations))
    for iteration in range(num_iterations + burnin):

        inside_sufficient_statistics = sample_inside_sufficient_statistics(model_parameters, noisy_inside_sufficient_statistics, noise_covariance, conjugate_pair, N, truncation_bounds)

        noise_covariance = sample_noise_covariance(inside_sufficient_statistics, noisy_inside_sufficient_statistics, epsilon, sensitivity)

        sufficient_statistics = sample_sufficient_statistics(model_parameters, noisy_inside_sufficient_statistics, noise_covariance, conjugate_pair, N, truncation_bounds)

        model_parameters, posterior_parameters = sample_model_parameters(sufficient_statistics, conjugate_pair, N)

        if iteration >= burnin:
            Gibbs_posterior[:, iteration - burnin] = model_parameters.flatten()

    return Gibbs_posterior


def initialize_Gibbs(conjugate_pair, epsilon, sensitivity):

    model_parameters = conjugate_pair.draw_model_parameters(conjugate_pair.prior_parameters)

    laplace_scale = 1.0 * sensitivity / epsilon

    noise_covariance = np.diag(np.random.exponential(scale=2 * laplace_scale ** 2, size=conjugate_pair.num_sufficient_statistics))

    return model_parameters, noise_covariance


def sample_inside_sufficient_statistics(model_parameters, noisy_sufficient_statistics, noise_covariance, conjugate_pair, N, truncation_bounds):

    # inside sufficient statistics parameters
    ss_mean, ss_variance = calculate_sufficient_statistics_section_parameters(N, model_parameters, conjugate_pair, truncation_bounds, 'inside')

    conditional_mean, conditional_covariance = product_of_two_multivariate_normals(noisy_sufficient_statistics, noise_covariance, ss_mean, ss_variance) # p(suff_stats | noisy_suff_stats, laplace_noise)

    sufficient_statistics = sample_constrained_sufficient_statistics_via_rejection_sampling(N, conditional_mean, conditional_covariance, conjugate_pair)

    return sufficient_statistics


def sample_sufficient_statistics(model_parameters, noisy_inside_sufficient_statistics, noise_covariance, conjugate_pair, N, truncation_bounds):

    # inside sufficient statistics parameters
    ss_mean_inside, ss_covariance_inside = calculate_sufficient_statistics_section_parameters(N, model_parameters, conjugate_pair, truncation_bounds, 'inside')
    ss_mean_inside, ss_covariance_inside = product_of_two_multivariate_normals(noisy_inside_sufficient_statistics, noise_covariance, ss_mean_inside, ss_covariance_inside)

    # lower sufficient statistics parameters
    ss_mean_lower, ss_covariance_lower = calculate_sufficient_statistics_section_parameters(N, model_parameters, conjugate_pair, truncation_bounds, 'lower')

    # upper sufficient statistics parameters
    ss_mean_upper, ss_covariance_upper = calculate_sufficient_statistics_section_parameters(N, model_parameters, conjugate_pair, truncation_bounds, 'upper')

    conditional_mean = ss_mean_lower + ss_mean_inside + ss_mean_upper
    conditional_covariance = ss_covariance_lower + ss_covariance_inside + ss_covariance_upper

    sufficient_statistics = sample_constrained_sufficient_statistics_via_rejection_sampling(N, conditional_mean, conditional_covariance, conjugate_pair)

    return sufficient_statistics


def sample_constrained_sufficient_statistics_via_rejection_sampling(N, conditional_mean, conditional_covariance, conjugate_pair):

    # copy to avoid rvs() changing shape of mean parameter
    conditional_mean = conditional_mean.copy()

    sufficient_statistics = scipy.stats.multivariate_normal.rvs(mean=conditional_mean, cov=conditional_covariance)

    # so that this variable is always a numpy array
    if isinstance(sufficient_statistics, float):
        sufficient_statistics = np.array([[sufficient_statistics]]).T

    tries = 0
    while not conjugate_pair.check_valid_sufficient_statistics(N, sufficient_statistics):

        sufficient_statistics = scipy.stats.multivariate_normal.rvs(mean=conditional_mean, cov=conditional_covariance)

        if isinstance(sufficient_statistics, float):
            sufficient_statistics = np.array([[sufficient_statistics]]).T

        tries += 1
        if tries > 100:
            raise ValueError('Model unable to sample sufficient statistics!')

    return sufficient_statistics[:, None]


def sample_noise_covariance(sufficient_statistics, noisy_inside_sufficient_statistics, epsilon, sensitivity):

    laplace_scale = 1.0 * sensitivity / epsilon

    abs_noise = np.abs(noisy_inside_sufficient_statistics - sufficient_statistics)

    inverse_variance = np.random.wald(1 / (laplace_scale * abs_noise), 1 / laplace_scale ** 2)

    covariance = 1 / inverse_variance

    if len(sufficient_statistics) > 1:
        covariance = np.diagflat(covariance)

    return covariance


def sample_model_parameters(sufficient_statistics, conjugate_pair, N):

    posterior_parameters = conjugate_pair.conjugate_update(N, conjugate_pair.prior_parameters, sufficient_statistics)

    model_parameters = conjugate_pair.draw_model_parameters(posterior_parameters)

    return model_parameters, posterior_parameters
