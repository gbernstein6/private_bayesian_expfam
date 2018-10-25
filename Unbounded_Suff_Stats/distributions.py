import scipy, scipy.stats
import numpy as np
import autograd.numpy as npa


'''
class Conjugate_Pair:

    def __init__(self):

        # let p = number of parameters used by likelihood model
        # let s = number of sufficient statistics used by likelihood model (note multinomial actually uses s-1)
        # let q = number of parameters used by prior model

        self.sensitivity = maximum amount addition/removal of an individual will change sufficient statistics

        self.prior_parameters = sequence of prior parameters to be used in self.draw_model_parameters(), shape (s,1)

        self.num_sufficient_statistics = number of feature functions for this model, to match length of self.draw_sufficient_statistics() return

    @staticmethod
    def draw_model_parameters(parameters, size=1):
        # parameters: prior parameters as type sequence
        # return: numpy array of shape (p,size)

    @staticmethod
    def draw_individual_data(N, model_parameters):
        # N: number of individuals in population (float or int)
        # model_parameters: numpy array of shape (p,1)
        # return: numpy array of shape (s,N)
        
    @staticmethod
    def model_ppf(model_parameters, q):
        # model_parameters: numpy array of shape (p,1)
        # q: quantile to calculate (float)
        # return: ppf (inverse cdf) of the model at q

    @staticmethod
    def get_feature_functions():
        # return: list of lambda functions, one for each feature function of the model

    @staticmethod
    def conjugate_update(N, prior_parameters, sufficient_statistics):
        # N: number of individuals in population (float or int)
        # prior_parameters: prior parameters as numpy array of shape (p,)
        # sufficient_statistics: numpy array of shape (s,1)
        # return: posterior parameters as numpy array of shape (q,)

    @staticmethod
    def check_valid_sufficient_statistics(N, sufficient_statistics):
        # N: number of individuals in population (float or int)
        # sufficient_statistics: numpy array of shape (s,1)
        # return: True if sufficient statistics are valid for the data model

    @staticmethod
    def patch_sufficient_statistics(N, sufficient_statistics):
        # N: number of individuals in population (float or int)
        # sufficient_statistics: numpy array of shape (s,1)
        # return: sufficient statistics projected to valid values

    ### NOTE: ### 
    The below methods are used in autograd and must be defined with autograd.numpy and autograd.scipy

    @staticmethod
    def model_to_natural_parameters(model_parameters):
        # model_parameters: numpy array of shape (p,1)
        # return: model parameters converted to natural parameterization of exponential family model

    @staticmethod
    def natural_to_model_parameters(natural_parameters):
        # natural_parameters: numpy array of shape (p,1)
        # return: natural parameters of exponential family model converted to model parameters 

    @staticmethod
    def cdf(x, model_parameters):
        # x: value at which to evaluate the cdf
        # model_parameters: numpy array of shape (p,1)
        # return: cdf of the model (float)

    @staticmethod
    def log_partition_function(natural_parameters):
        # natural_parameters: numpy array of shape (p,1)
        # return: log partition function of the model (float)
'''


class Gamma_Exponential:

    def __init__(self):

        alpha = 8.0
        beta = 2.0
        self.prior_parameters = np.array([[alpha, beta]]).T  # [shape, 1/scale]

        self.num_sufficient_statistics = 1

    @staticmethod
    def draw_model_parameters(parameters, size=1):
        return scipy.stats.gamma.rvs(parameters[0, 0], scale=1.0/parameters[1, 0], size=size)[:, None].T

    @staticmethod
    def draw_individual_data(N, model_parameters):

        individual_data = scipy.stats.expon.rvs(scale=1.0/model_parameters[0], size=int(N))[:, None].T

        return individual_data

    @staticmethod
    def model_ppf(model_parameters, q):
        return scipy.stats.expon.ppf(q, scale=1.0/model_parameters[0])

    @staticmethod
    def get_feature_functions():
        return [lambda x: x]

    @staticmethod
    def conjugate_update(N, prior_parameters, sufficient_statistics):

        posterior_parameters = np.array([[prior_parameters[0, 0] + N,
                                          prior_parameters[1, 0] + np.max((sufficient_statistics, .01))]]).T

        return posterior_parameters

    @staticmethod
    def check_valid_sufficient_statistics(N, sufficient_statistics):

        # ensure non-negative value
        return sufficient_statistics[0] >= 0

    @staticmethod
    def patch_sufficient_statistics(N, sufficient_statistics):

        # ensure non-negative sum(x)
        sufficient_statistics[0] = np.max((sufficient_statistics[0], 0))

        return sufficient_statistics

    @staticmethod
    def model_to_natural_parameters(model_parameters):
        return - model_parameters

    @staticmethod
    def natural_to_model_parameters(natural_parameters):
        return - natural_parameters

    @staticmethod
    def cdf(x, model_parameters):
        return 1 - npa.exp(-model_parameters[0] * x).squeeze()

    @staticmethod
    def log_partition_function(natural_parameters):
        return -npa.log(-natural_parameters).squeeze()
