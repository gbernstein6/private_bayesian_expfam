from autograd import numpy as npa
from autograd import grad, hessian

import driver


def calculate_sufficient_statistics_section_parameters(N, model_parameters, conjugate_pair, truncation_bounds, section):
    # section is one of ['lower', 'inside', 'upper']

    natural_parameters = conjugate_pair.model_to_natural_parameters(model_parameters)

    feature_function_mean = calculate_expected_feature_function(natural_parameters, truncation_bounds, section)
    feature_function_covariance = calculate_feature_function_covariance(natural_parameters, truncation_bounds, section)

    # now calculate suff stat params
    truncated_cdf = calculate_truncated_cdf(conjugate_pair, model_parameters, truncation_bounds, section)

    sufficient_statistics_mean = N * truncated_cdf * feature_function_mean

    ss_covariance = N * truncated_cdf * feature_function_covariance \
                    + N * truncated_cdf * (1 - truncated_cdf) * npa.dot(feature_function_mean, feature_function_mean.T)

    return sufficient_statistics_mean, ss_covariance


# expected sufficient statistics
def calculate_expected_feature_function(natural_parameters, truncation_bounds, section):
    return grad(log_partition_function_truncated)(natural_parameters, truncation_bounds, section)


# variance sufficient statistics
def calculate_feature_function_covariance(natural_parameters, truncation_bounds, section):
    return hessian(log_partition_function_truncated)(natural_parameters.flatten(), truncation_bounds, section)


# Define the truncated log-partition function using our version of normcdf
def log_partition_function_truncated(natural_parameters, truncation_bounds, section):

    conjugate_pair = driver.conjugate_pair

    model_parameters = conjugate_pair.natural_to_model_parameters(natural_parameters)

    truncated_cdf = calculate_truncated_cdf(conjugate_pair, model_parameters, truncation_bounds, section)

    return conjugate_pair.log_partition_function(natural_parameters) + npa.log(truncated_cdf)


def calculate_truncated_cdf(conjugate_pair, model_parameters, truncation_bounds, section):

    lower_cdf = conjugate_pair.cdf(truncation_bounds[0], model_parameters)
    upper_cdf = conjugate_pair.cdf(truncation_bounds[1], model_parameters)

    if section == 'lower':
        truncated_cdf = lower_cdf
    elif section == 'inside':
        truncated_cdf = upper_cdf - lower_cdf
    elif section == 'upper':
        truncated_cdf = 1 - upper_cdf
    else:
        raise ValueError('Unrecognized section! (%s)' % section)

    return truncated_cdf
