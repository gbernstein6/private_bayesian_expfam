import numpy as np


def generate_noisy_sufficient_statistics(N, epsilon, conjugate_pair, model_parameters):

    individual_data = conjugate_pair.draw_individual_data(N, model_parameters)

    truncation_bounds = determine_truncation_bounds(conjugate_pair, model_parameters)

    feature_functions = conjugate_pair.get_feature_functions()

    sensitivity = calculate_sensitivity(conjugate_pair, truncation_bounds)

    sufficient_statistics = np.array([np.sum([formula(x) for x in individual_data]) for formula in feature_functions])[:, None]

    inside_individual_data = individual_data[np.logical_and(truncation_bounds[0] < individual_data, individual_data < truncation_bounds[1])]
    inside_sufficient_statistics = np.array([np.sum([formula(x) for x in inside_individual_data]) for formula in feature_functions])[:, None]

    noise = draw_laplace_noise(sensitivity, epsilon, sufficient_statistics.shape)

    noisy_sufficient_statistics = sufficient_statistics + noise
    noisy_inside_sufficient_statistics = inside_sufficient_statistics + noise

    return sufficient_statistics, noisy_sufficient_statistics, noisy_inside_sufficient_statistics, truncation_bounds


def determine_truncation_bounds(conjugate_pair, model_parameters, percent_to_cut=.05):

    lower_trunc = conjugate_pair.model_ppf(model_parameters, percent_to_cut / 2.0)[0]
    upper_trunc = conjugate_pair.model_ppf(model_parameters, 1 - percent_to_cut / 2.0)[0]

    return lower_trunc, upper_trunc


def calculate_sensitivity(conjugate_pair, truncation_bounds):

    feature_functions = conjugate_pair.get_feature_functions()

    sensitivity = np.sum([np.abs(formula(truncation_bounds[1]) - formula(truncation_bounds[0]))
                          for formula in feature_functions])

    return sensitivity


def draw_laplace_noise(sensitivity, epsilon, shape):

    laplace_scale = 1.0 * sensitivity / epsilon

    laplace_noise = np.random.laplace(scale=laplace_scale, size=shape)

    return laplace_noise


def product_of_two_multivariate_normals(mean1, cov1, mean2, cov2):

    # https://math.stackexchange.com/questions/157172/product-of-two-multivariate-gaussians-distributions
    temp = np.linalg.inv(cov1 + cov2)
    combined_covariance = cov1.dot(temp).dot(cov2)
    combined_mean = cov2.dot(temp).dot(mean1) \
                       + cov1.dot(temp).dot(mean2)

    return combined_mean, combined_covariance


def calculate_conjugate_update_posterior(sufficient_statistics, num_samples, conjugate_pair, N):

    patched_sufficient_statistics = conjugate_pair.patch_sufficient_statistics(N, sufficient_statistics)

    posterior_parameters = conjugate_pair.conjugate_update(N, conjugate_pair.prior_parameters, patched_sufficient_statistics)

    posterior_samples = conjugate_pair.draw_model_parameters(posterior_parameters, size=num_samples)

    return posterior_samples
