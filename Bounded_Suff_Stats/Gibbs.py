import numpy as np


def calculate_Gibbs_posterior(noisy_sufficient_statistics, conjugate_pair, N, epsilon):
    # returns posterior as numpy array of shape (p, i), where p is number of model parameters and i is number of Gibbs samples

    burnin, num_iterations = 500, 1000

    model_parameters, noise_covariance = initialize_Gibbs(conjugate_pair, epsilon)

    Gibbs_posterior = np.zeros((len(model_parameters), num_iterations))
    for iteration in range(num_iterations + burnin):

        sufficient_statistics = sample_sufficient_statistics(model_parameters, noisy_sufficient_statistics, noise_covariance, conjugate_pair, N)

        noise_covariance = sample_noise_covariance(sufficient_statistics, noisy_sufficient_statistics, conjugate_pair, epsilon)

        model_parameters, posterior_parameters = sample_model_parameters(sufficient_statistics, conjugate_pair, N)

        if iteration >= burnin:
            Gibbs_posterior[:, iteration - burnin] = model_parameters.flatten()

    return Gibbs_posterior


def initialize_Gibbs(conjugate_pair, epsilon):

    model_parameters = conjugate_pair.draw_model_parameters(conjugate_pair.prior_parameters)

    laplace_scale = 1.0 * conjugate_pair.sensitivity / epsilon

    exp_scale = 2 * laplace_scale ** 2

    noise_covariance = np.diag(np.random.exponential(scale=exp_scale, size=conjugate_pair.num_sufficient_statistics))

    return model_parameters, noise_covariance


def sample_sufficient_statistics(model_parameters, noisy_sufficient_statistics, noise_covariance, conjugate_pair, N):

    ss_mean, ss_covariance = conjugate_pair.calculate_sufficient_statistics_CLT_parameters(N, model_parameters)

    # p(suff_stats | noisy_suff_stats, laplace_noise)
    conditional_mean, conditional_covariance = product_of_two_multivariate_normals(noisy_sufficient_statistics, noise_covariance,
                                                                                   ss_mean, ss_covariance)

    sufficient_statistics = conjugate_pair.sample_sufficient_statistics(N, conditional_mean, conditional_covariance)

    return sufficient_statistics


def product_of_two_multivariate_normals(mean1, cov1, mean2, cov2):

    # https://math.stackexchange.com/questions/157172/product-of-two-multivariate-gaussians-distributions
    temp = np.linalg.inv(cov1 + cov2)
    combined_covariance = cov1.dot(temp).dot(cov2)
    combined_mean = cov2.dot(temp).dot(mean1) \
                       + cov1.dot(temp).dot(mean2)

    return combined_mean, combined_covariance


def sample_noise_covariance(sufficient_statistics, noisy_sufficient_statistics, conjugate_pair, epsilon):

    laplace_scale = 1.0 * conjugate_pair.sensitivity / epsilon

    abs_noise = np.abs(noisy_sufficient_statistics - sufficient_statistics)

    inverse_covariance = np.random.wald(1 / (laplace_scale * abs_noise), 1 / laplace_scale ** 2)

    covariance = np.diagflat(1 / inverse_covariance)

    return covariance


def sample_model_parameters(sufficient_statistics, conjugate_pair, N):

    posterior_parameters = conjugate_pair.conjugate_update(N, conjugate_pair.prior_parameters, sufficient_statistics)

    model_parameters = conjugate_pair.draw_model_parameters(posterior_parameters)

    return model_parameters, posterior_parameters
