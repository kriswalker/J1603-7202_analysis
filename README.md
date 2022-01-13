Here you will find the scripts for reproducing the results reported in "Orbital dynamics and extreme scattering event properties from long-term scintillation observations of PSR J1603-7202", Walker et al. (2022). The code requires [scintools](https://github.com/danielreardon/scintools).

The `.npz` files containing the curvature profile data used in our analysis are stored in `data/curvature_profile_data`. The full set of raw dynamic spectra are available for download from the CSIRO data access portal and can be converted into the input profile data using the `save_curvature_data` method in `scintools.scint_utils`.

Below is a list of the included `.py` files, with a short description of their purpose and usage:

* `model_J1603.py`: script for Bayesian modelling with the static (1 epoch) model. It reads in the data, defines the priors, and runs the inference. The number of initial points and other dynesty sampling options can be modified inside the script, as can the anisotropy option and output directory. Outputs a results file, a list of samples, and the corner plot.

* `model_J1603_tvary.py`: script for Bayesian modelling with the multiple-epoch models. Does the same thing as `model_J1603.py` but with 2-5 epochs, chosen by modifying the `nepochs` option in the script. Adding more epochs simply requires adding additional `eta_model_*` functions and corresponding `if` statements below the prior definitions.

* `read_curvature_data.py`: reads in curvature profile data, averages simultaneous observations, and performs further processing as described in Section 3.2 of the paper.

* `bilby_likelihood.py`: contains bilby-compatible wrapper classes for the likelihood function. Calls the `curvature_log_likelihood` function included in `scintools.scint_utils`.

* `plot_J1603.py`: script for generating plots of the data and model prediction (Figure 5 of the paper) and pp-plots (Figure 6).

* `bilby_plotting.py`: functions called by `plot_J1603.py`.

* `differential_rotation.py`: calculates the contribution to the velocity from differential rotation of the galaxy, as described in Section 6.2 of the paper.

For questions and comments, please contact the corresponding author at kris.walker.astro@gmail.com.
