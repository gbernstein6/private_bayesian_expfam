import scipy, scipy.stats
import numpy as np

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
        # parameters: (prior or posterior) parameters as numpy array of shape (p,1)
        # return: numpy array of shape (p,size)

    @staticmethod
    def draw_sufficient_statistics(N, model_parameters):
        # N: number of individuals in population (float or int)
        # model_parameters: sequence returned by self.draw_model_parameters()
        # return: numpy array of shape (s,1)

    @staticmethod
    def conjugate_update(N, prior_parameters, sufficient_statistics):
        # N: number of individuals in population (float or int)
        # prior_parameters: prior parameters as numpy array of shape (p,)
        # sufficient_statistics: numpy array of shape (s,1)
        # return: posterior parameters as numpy array of shape (q,)

    @staticmethod
    def calculate_sufficient_statistics_CLT_parameters(N, model_parameters):
        # N: number of individuals in population (float or int)
        # model_parameters: parameters as numpy array of shape (p,)
        # return: ss_mean: numpy array of shape (s,1)
        #         ss_covariance: numpy array of shape (s,s)

    @staticmethod
    def sample_sufficient_statistics(N, conditional_mean, conditional_covariance):
        # N: number of individuals in population (float or int)
        # conditional mean: numpy array of shape (s,1)
        # conditional_covariance: numpy array of shape (s,s)
        # return: sufficient_statistics: numpy array of shape (s,1)
'''


class Beta_Binomial:

    def __init__(self):

        self.sensitivity = 1

        self.prior_parameters = np.array([[10, 10]]).T  # [alpha, beta]

        self.num_sufficient_statistics = 1  # one binomial parameter

    @staticmethod
    def draw_model_parameters(parameters, size=1):
        return scipy.stats.beta.rvs(parameters[0, 0], parameters[1, 0], size=size)[:, None].T

    @staticmethod
    def draw_sufficient_statistics(N, model_parameters):
        return np.array([scipy.stats.binom.rvs(int(N), model_parameters)])[:, None]

    @staticmethod
    def conjugate_update(N, prior_parameters, sufficient_statistics):

        posterior_parameters = np.array([prior_parameters[0, 0] + sufficient_statistics,
                                         prior_parameters[1, 0] + N - sufficient_statistics])

        # make sure beta parameters are positive
        posterior_parameters = np.maximum(posterior_parameters, .001)

        return posterior_parameters

    @staticmethod
    def calculate_sufficient_statistics_CLT_parameters(N, model_parameters):

        ss_mean = np.array([N * model_parameters])
        ss_covariance = np.array([N * model_parameters * (1 - model_parameters)])

        return ss_mean, ss_covariance

    @staticmethod
    def sample_sufficient_statistics(N, conditional_mean, conditional_covariance):

        conditional_std = np.sqrt(conditional_covariance)

        # draw sufficient statistics constrained to [0, N]
        a = (0 - conditional_mean) / conditional_std
        b = (N - conditional_mean) / conditional_std

        sufficient_statistics = np.array([[scipy.stats.truncnorm.rvs(a, b, loc=conditional_mean, scale=conditional_std)]])

        return sufficient_statistics


class Dirichlet_Multinomial:
    # NOTE: If the model has M parameters, then we only carry around M-1 parameters and sufficient statistics
    #       so that the parameters can sum to 1 and sufficient statistics can sum to N
    # The last value is only ever added into the sufficient statistics when drawing model parameters

    def __init__(self):

        self.sensitivity = 1

        self.prior_parameters = np.ones((3, 1)) * 5

        self.num_sufficient_statistics = len(self.prior_parameters) - 1

    @staticmethod
    def draw_model_parameters(parameters, size=1):

        model_parameters = scipy.stats.dirichlet.rvs(parameters.flatten(), size=size).T

        # only carry around s-1 parameters
        return model_parameters[:-1, :]

    @staticmethod
    def draw_sufficient_statistics(N, model_parameters):

        # add last parameter back in to sum to 1
        model_parameters = np.vstack((model_parameters, 1.0 - sum(model_parameters)))

        sufficient_statistics = np.array([float(x) for x in np.random.multinomial(N, model_parameters.flatten())])[:, None]

        # only carry around s-1 sufficient statistics
        sufficient_statistics = sufficient_statistics[:-1]

        return sufficient_statistics

    @staticmethod
    # first parameter N is unneeded
    def conjugate_update(N, prior_parameters, sufficient_statistics):

        sufficient_statistics = np.vstack((sufficient_statistics, N - sum(sufficient_statistics)))

        posterior_parameters = prior_parameters + sufficient_statistics

        return posterior_parameters

    @staticmethod
    def calculate_sufficient_statistics_CLT_parameters(N, model_parameters):

        ss_mean = N * model_parameters

        ss_covariance = - N * model_parameters.dot(model_parameters.T)
        np.fill_diagonal(ss_covariance, N * model_parameters * (1 - model_parameters))

        return ss_mean, ss_covariance

    @staticmethod
    def sample_sufficient_statistics(N, conditional_mean, conditional_covariance):

        sufficient_statistics = scipy.stats.multivariate_normal.rvs(mean=conditional_mean.flatten(), cov=conditional_covariance, size=1)

        # ensure positive values and that at least one count is left over for the tacked on value
        tries = 0
        while not all([ss > 0 for ss in sufficient_statistics[:-1]]) and sum(sufficient_statistics[:-1]) <= N - 1:
            sufficient_statistics = scipy.stats.multivariate_normal.rvs(mean=conditional_mean.flatten(), cov=conditional_covariance, size=1)

            if tries > 100:
                raise Exception('Multinomial unable to sample sufficient statistics!')

        return sufficient_statistics[:, None]
