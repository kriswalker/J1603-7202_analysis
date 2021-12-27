import numpy as np
from scintools.scint_utils import read_par, get_earth_velocity, \
    pars_to_params, get_true_anomaly
from scintools.scint_models import arc_curvature
from bilby.core import result
from read_curvature_data import read_data
from bilby_plotting import custom_plot_with_data, pp_plot

average = True
anisotropy = True
results_dir = 'results/outdir_anisotropic/'
results_file = results_dir + 'dynesty_result.json'
outdir = results_dir
filename = 'plot_with_data.png'
plot_pp = True

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

##############################################################################

vearth_ra, vearth_dec = get_earth_velocity(mjd_arc, pars['RAJ'], pars['DECJ'])
true_anomaly = get_true_anomaly(mjd_arc, params)

true_anomaly = true_anomaly.squeeze()
vearth_ra = vearth_ra.squeeze()
vearth_dec = vearth_dec.squeeze()

ninterp = 10000
print('Generating interpolated Earth velocity and true anomaly for plot')
mjd_arc_ = np.linspace(np.min(mjd_arc), np.max(mjd_arc), ninterp)
vearth_ra_, vearth_dec_ = get_earth_velocity(mjd_arc_, pars['RAJ'],
                                             pars['DECJ'])
true_anomaly_ = get_true_anomaly(mjd_arc_, params)

true_anomaly_ = true_anomaly_.squeeze()
vearth_ra_ = vearth_ra_.squeeze()
vearth_dec_ = vearth_dec_.squeeze()


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
                           true_anomaly_, vearth_ra_, vearth_dec_)

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

    def setter(ta, vra, vdec, ta_, vra_, vdec_):
        if np.shape(xdata) == np.shape(mjd_arc_):
            return [ta_, vra_, vdec_]
        else:
            return [ta, vra, vdec]

    var = setter(true_anomaly, vearth_ra, vearth_dec,
                 true_anomaly_, vearth_ra_, vearth_dec_)
    ydata = np.zeros(np.shape(xdata))
    weights = np.ones(np.shape(xdata))

    model = -arc_curvature(params_, ydata, weights,
                           var[0], var[1], var[2])

    return model


if anisotropy:
    eta_model = eta_model_anisotropic
else:
    eta_model = eta_model_isotropic

results = result.read_in_result(filename=results_file, outdir=None, label=None,
                                extension='json', gzip=False)
ndraws = 1000
custom_plot_with_data(results, eta_model, mjd_arc, nfdop, power, noise,
                      ydraws=100, ndraws=ndraws,
                      draws_label='{} draws'.format(ndraws), npoints=ninterp,
                      xlabel='MJD', ylabel=r'1/$\sqrt{\eta}$',
                      outfile=outdir+filename, modulo=True, log=False,
                      params=params, plot_residuals=False, ylimits=(0, 6e-2),
                      xlimits=(0, 1), save=True)
if plot_pp:
    pp_plot(results, eta_model, mjd_arc, nfdop, power, noise, ndraws=2,
            outfile=outdir+'pp_plot.png')
