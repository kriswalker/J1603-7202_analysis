"""
J1603-7202 arc curvature modelling with static screen parameters
"""

import numpy as np
from scintools.scint_utils import read_par, get_earth_velocity, \
    pars_to_params, get_true_anomaly
from scintools.scint_models import arc_curvature
import bilby
from bilby_likelihood import CurvatureLikelihood
from read_curvature_data import read_data

"""
Options
"""

npoints = 1000
average = True
anisotropy = True
outfile = 'results/outdir_anisotropic'

print('Average obs    = {0}\
       \nAnisotropic    = {1}\n'.format(average, anisotropy))

"""
Read results and set parameter arrays
"""
pars = read_par('data/J1603-7202.par')
params = pars_to_params(pars)

datapath = 'data/curvature_profile_data/*'

data, data_, ids = read_data(datapath,
                             average_obs=average,
                             sel_dir=None,
                             discard='data/selection/discard.txt',
                             average='data/selection/average.txt'
                             )

nfdop = data[0]
power = data[1]
noise = data[2]
mjd_arc = data[3]
names_arc = data[4]

nfdop_ = data_[0]
power_ = data_[1]
noise_ = data_[2]
mjd_arc_ = data_[3]
names_arc_ = data_[4]


"""
Model the curvature
"""
print('Getting Earth velocity')
vearth_ra, vearth_dec = get_earth_velocity(mjd_arc, pars['RAJ'], pars['DECJ'])

print('Getting true anomaly')
true_anomaly = get_true_anomaly(mjd_arc, params)

true_anomaly = true_anomaly.squeeze()
vearth_ra = vearth_ra.squeeze()
vearth_dec = vearth_dec.squeeze()


def eta_model_isotropic(xdata, cosi, kom, s, d, vism_ra, vism_dec, efac,
                        equad):
    """
    bilby-compatible function for calling arc curvature model
    """
    params_ = dict(params)

    params_['COSI'] = cosi
    params_['KOM'] = kom
    params_['s'] = s
    params_['d'] = d

    params_['vism_ra'] = vism_ra
    params_['vism_dec'] = vism_dec

    ydata = np.zeros(np.shape(xdata))
    weights = np.ones(np.shape(xdata))

    model = -arc_curvature(params_, ydata, weights,
                           true_anomaly, vearth_ra, vearth_dec)

    return model


def eta_model_anisotropic(xdata, cosi, kom, s, d, psi, vism_psi, efac, equad):
    """
    bilby-compatible function for calling arc curvature model
    """
    params_ = dict(params)

    params_['COSI'] = cosi
    params_['KOM'] = kom
    params_['s'] = s
    params_['d'] = d

    params_['psi'] = psi
    params_['vism_psi'] = vism_psi

    ydata = np.zeros(np.shape(xdata))
    weights = np.ones(np.shape(xdata))

    model = -arc_curvature(params_, ydata, weights,
                           true_anomaly, vearth_ra, vearth_dec)

    return model


priors = dict()
priors['cosi'] = bilby.core.prior.Uniform(-1, 1, 'cosi')
priors['kom'] = bilby.core.prior.Uniform(0, 360, 'kom', boundary='periodic')
priors['s'] = bilby.core.prior.Uniform(0, 1, 's')
priors['d'] = bilby.core.prior.TruncatedGaussian(3.4, 0.5, 0, 100, 'd')
if anisotropy:
    priors['psi'] = bilby.core.prior.Uniform(0, 180, 'psi',
                                             boundary='periodic')
    priors['vism_psi'] = bilby.core.prior.Gaussian(0, 100, 'vism_psi')
else:
    priors['vism_ra'] = bilby.core.prior.Gaussian(0, 100, 'vism_ra')
    priors['vism_dec'] = bilby.core.prior.Gaussian(0, 100, 'vism_dec')
priors['efac'] = bilby.core.prior.Uniform(-np.log(5), np.log(20), 'efac')
priors['equad'] = bilby.core.prior.Uniform(0, 2, 'equad')

if anisotropy:
    eta_model = eta_model_anisotropic
else:
    eta_model = eta_model_isotropic
likelihood = CurvatureLikelihood(mjd_arc, eta_model, nfdop, power, noise)

results = bilby.core.sampler.run_sampler(likelihood, priors=priors,
                                         sampler='dynesty', label='dynesty',
                                         npoints=npoints, verbose=False,
                                         resume=False, check_point=True,
                                         check_point_delta_t=600,
                                         outdir=outfile)
results.plot_corner()
