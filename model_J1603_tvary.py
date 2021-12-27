"""
J1603 arc curvature modelling for multiple epochs
"""

import numpy as np
from scintools.scint_utils import read_par, get_earth_velocity, \
                                    pars_to_params, get_true_anomaly
from scintools.scint_models import arc_curvature
import bilby
from bilby_likelihood import CurvatureLikelihoodEpochs
from read_curvature_data import read_data

"""
Options
"""

npoints = 1000
nepochs = 3
outfile = 'results/outdir_anisotropic_{}epochs'.format(nepochs)
average = True
anisotropy = True

print('No. of epochs  = {0}\
       \nAverage obs    = {1}\
       \nAnisotropic    = {2}\n'.format(nepochs, average, anisotropy))

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

period_mjd = np.max(mjd_arc) - np.min(mjd_arc)


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


def calc_model(p, x, psi_array, vism_psi_array, s_array, epoch_array):

    ydata = np.zeros(np.shape(x))
    weights = np.ones(np.shape(x))

    m = np.array([])
    mjd_min = np.min(x)
    for i in range(nepochs):
        p['psi'] = psi_array[i]
        p['vism_psi'] = vism_psi_array[i]
        p['s'] = s_array[i]

        start = epoch_array[i] + mjd_min
        end = epoch_array[i+1] + mjd_min
        epoch_ind = np.argwhere((x >= start) &
                                (x < end))
        if len(epoch_ind) != 0:
            mi = -arc_curvature(p, ydata[epoch_ind].flatten(),
                                weights[epoch_ind].flatten(),
                                true_anomaly[epoch_ind].flatten(),
                                vearth_ra[epoch_ind].flatten(),
                                vearth_dec[epoch_ind].flatten())
            m = np.append(m, mi)
    m = m.flatten()

    return m


def eta_model_2(xdata, cosi, kom, d, s1, s2, psi1, psi2, vism_psi1, vism_psi2,
                efac, equad, epoch1):

    params_ = dict(params)

    params_['COSI'] = cosi
    params_['KOM'] = kom
    params_['d'] = d

    psi_arr = [psi1, psi2]
    vism_psi_arr = [vism_psi1, vism_psi2]
    s_arr = [s1, s2]
    epoch_ends = [-1e9, epoch1, 1e9]

    return calc_model(params_, xdata, psi_arr, vism_psi_arr, s_arr, epoch_ends)


def eta_model_3(xdata, cosi, kom, d, s1, s2, s3, psi1, psi2, psi3, vism_psi1,
                vism_psi2, vism_psi3, efac, equad, epoch1, epoch2):

    params_ = dict(params)

    params_['COSI'] = cosi
    params_['KOM'] = kom
    params_['d'] = d

    psi_arr = [psi1, psi2, psi3]
    vism_psi_arr = [vism_psi1, vism_psi2, vism_psi3]
    s_arr = [s1, s2, s3]
    epoch_ends = [-1e9, epoch1, epoch2, 1e9]

    return calc_model(params_, xdata, psi_arr, vism_psi_arr, s_arr, epoch_ends)


def eta_model_4(xdata, cosi, kom, d, s1, s2, s3, s4, psi1, psi2, psi3, psi4,
                vism_psi1, vism_psi2, vism_psi3, vism_psi4, efac, equad,
                epoch1, epoch2, epoch3):

    params_ = dict(params)

    params_['COSI'] = cosi
    params_['KOM'] = kom
    params_['d'] = d

    psi_arr = [psi1, psi2, psi3, psi4]
    vism_psi_arr = [vism_psi1, vism_psi2, vism_psi3, vism_psi4]
    s_arr = [s1, s2, s3, s4]
    epoch_ends = [-1e9, epoch1, epoch2, epoch3, 1e9]

    return calc_model(params_, xdata, psi_arr, vism_psi_arr, s_arr, epoch_ends)


def eta_model_5(xdata, cosi, kom, d, s1, s2, s3, s4, s5, psi1, psi2, psi3,
                psi4, psi5, vism_psi1, vism_psi2, vism_psi3, vism_psi4,
                vism_psi5, efac, equad, epoch1, epoch2, epoch3, epoch4):

    params_ = dict(params)

    params_['COSI'] = cosi
    params_['KOM'] = kom
    params_['d'] = d

    psi_arr = [psi1, psi2, psi3, psi4, psi5]
    vism_psi_arr = [vism_psi1, vism_psi2, vism_psi3, vism_psi4, vism_psi5]
    s_arr = [s1, s2, s3, s4, s5]
    epoch_ends = [-1e9, epoch1, epoch2, epoch3, epoch4, 1e9]

    return calc_model(params_, xdata, psi_arr, vism_psi_arr, s_arr, epoch_ends)


priors = dict()
priors['cosi'] = bilby.core.prior.Uniform(-1, 1, 'cosi')
priors['kom'] = bilby.core.prior.Uniform(0, 360, 'kom', boundary='periodic')
priors['d'] = bilby.core.prior.TruncatedGaussian(3.4, 0.5, 0, 100, 'd')
priors['efac'] = bilby.core.prior.Uniform(-np.log(5), np.log(20), 'efac')
priors['equad'] = bilby.core.prior.Uniform(0, 2, 'equad')
for i in range(1, nepochs+1):
    priors['s{}'.format(i)] = bilby.core.prior.Uniform(0, 1, 's{}'.format(i))
    priors['psi{}'.format(i)] = bilby.core.prior.Uniform(0, 180,
                                                         'psi{}'.format(i),
                                                         boundary='periodic')
    priors['vism_psi{}'.format(i)] = \
        bilby.core.prior.Gaussian(0, 100, 'vism_psi{}'.format(i))
    if i < nepochs:
        priors['epoch{}'.format(i)] = \
            bilby.core.prior.Uniform(0, period_mjd, 'epoch{}'.format(i))

if nepochs == 2:
    eta_model = eta_model_2
if nepochs == 3:
    eta_model = eta_model_3
if nepochs == 4:
    eta_model = eta_model_4
if nepochs == 5:
    eta_model = eta_model_5

likelihood = CurvatureLikelihoodEpochs(mjd_arc, eta_model, nfdop,
                                       power, noise, nepochs)

results = bilby.core.sampler.run_sampler(likelihood, priors=priors,
                                         sampler='dynesty', label='dynesty',
                                         npoints=npoints, verbose=False,
                                         resume=False, check_point=True,
                                         check_point_delta_t=600,
                                         outdir=outfile)
results.plot_corner()
