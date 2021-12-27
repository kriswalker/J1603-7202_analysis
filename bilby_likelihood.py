import numpy as np
import bilby
from scintools.scint_utils import curvature_log_likelihood


class CurvatureLikelihood(bilby.likelihood.Analytical1DLikelihood):
    def __init__(self, x, func, eta, power, noise):

        super(CurvatureLikelihood, self).__init__(x=x, y=np.zeros(np.shape(x)),
                                                  func=func)

        self.norm_eta = eta
        self.power = power
        self.noise = noise

    def log_likelihood(self):
        efac = self.model_parameters['efac']
        equad = self.model_parameters['equad']
        self.sigma = np.sqrt((self.noise * np.exp(efac))**2 + equad**2)

        # convert model eta to normalized value
        ymodel = 10/np.sqrt(-self.residual)

        return curvature_log_likelihood(self.power, self.norm_eta, self.sigma,
                                        ymodel)


class CurvatureLikelihoodEpochs(bilby.likelihood.Analytical1DLikelihood):
    def __init__(self, x, func, eta, power, noise, nepochs):

        super(CurvatureLikelihoodEpochs, self).__init__(
            x=x, y=np.zeros(np.shape(x)), func=func)

        self.norm_eta = eta
        self.power = power
        self.noise = noise
        self.nepochs = nepochs

    def log_likelihood(self):
        epoch_arr = [self.model_parameters['epoch{}'.format(x)]
                     for x in range(1, self.nepochs)]
        if np.all(np.diff(epoch_arr) > 0):
            efac = self.model_parameters['efac']
            equad = self.model_parameters['equad']
            self.sigma = np.sqrt((self.noise * np.exp(efac))**2 + equad**2)

            # convert model eta to normalized value
            ymodel = 10/np.sqrt(-self.residual)

            return curvature_log_likelihood(self.power, self.norm_eta,
                                            self.sigma, ymodel)
        else:
            return -np.inf
