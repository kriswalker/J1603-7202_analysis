import numpy as np
import matplotlib.pyplot as plt
from scintools.scint_utils import calculate_curvature_peak_probability
import bilby
from collections import OrderedDict

def custom_plot_with_data(result, model, x, eta, power, noise, ydraws=20, 
                           ndraws=1000, npoints=1000,
                           xlabel=None, ylabel=None, data_label='data',
                           residuals_label='residuals',
                           data_fmt='o', draws_label=None, filename=None,
                           maxl_label='max likelihood', dpi=300, outfile=None, 
                           modulo=True, log=True,
                           params=None, xlimits=None, ylimits=None,
                           save=True, plot_residuals=True,
                           figsize=(8,5)):

    model_keys = bilby.utils.infer_parameters_from_function(model)
    model_posterior = result.posterior[model_keys]
    
    period = params['PB'].value
    xsmooth = np.linspace(np.min(x), np.max(x), npoints)
    phase_smooth = (xsmooth % period) / period
    phase = (x % period) / period
    if modulo:
        x_orig = x
        x = phase
        xlabel = 'Orbital phase'
    fig = plt.figure(figsize=figsize, dpi=200)
    if plot_residuals:
        loc = 211
    else:
        loc = 111
    ax = fig.add_subplot(loc)
    print('Plotting {} draws'.format(ndraws))
    for _ in range(ndraws):
        s = model_posterior.sample().to_dict('records')[0]
        if _ > 0:
            draws_label = None
        ax.plot(phase_smooth,
                1/np.sqrt(model(xsmooth, **s)),
                data_fmt, alpha=0.25, lw=0.1, color='r',
                label=draws_label, linestyle='', markersize=0.5)
    try:
        if all(~np.isnan(result.posterior.log_likelihood)):
            print('Plotting maximum likelihood')
            s_maxl = model_posterior.iloc[result.posterior.log_likelihood.idxmax()]
            ax.plot(phase_smooth,
                    1/np.sqrt(model(xsmooth, **s_maxl)), 
                    data_fmt, lw=1, color='k',
                    label=maxl_label, linestyle='', markersize=0.5)
    except (AttributeError, TypeError):
        print("No log likelihood values stored, unable to plot max")
            
    y = np.zeros((len(eta), ydraws))
    ymax = np.zeros(len(eta))
    eta_stats = []
    
    sigmasq = (np.exp(s_maxl['efac']) * noise)**2 + s_maxl['equad']**2
    prob = calculate_curvature_peak_probability(power, np.sqrt(sigmasq))
                  
    for i,p in enumerate(prob):
        maxind = np.argmax(p)
        ymax[i] = eta[i,maxind]
        fac = ydraws / np.max(p)
        pscale = p * fac
        eta_ = []
        for j,c in enumerate(eta[i]):
            eta_ += [c] * int(round(pscale[j]))
        eta_stats.append(np.array(eta_) / 10)
    eta_stats = np.array(eta_stats)
    
    y = 1e2 / y**2
    ymax = 1e2 / ymax**2
            
    y = 1/np.sqrt(y)
    ymax = 1/np.sqrt(ymax)
        
    def plot_violin(axes, stats, xdata, color='C0', label='data'):
        parts = axes.violinplot(stats, xdata, 
                              widths=0.01, showextrema=False)
        for pc in parts['bodies']:
            pc.set_alpha(0.5)
            pc.set_zorder(10)
        plt.axhspan(1, 2, color=color, alpha=0.5, label=label)
    
    plot_violin(ax, eta_stats, x, color='C0', label=data_label)
            
    if plot_residuals:
        m = []
        nres = 1
        for _ in range(nres):    
            m.append(1 / np.sqrt(model(x_orig, **s_maxl)))
        m = np.array(m)
        ax_res = fig.add_subplot(212)
        eta_stats_res = []
        for i in range(len(eta_stats)):
            eta_stats_i = eta_stats[i]
            res = []
            for j in range(nres):
                res.append(eta_stats_i - m[j, i])
            res = np.array(res).flatten()
            eta_stats_res.append(res)
        eta_stats_res = np.array(eta_stats_res)
        plot_violin(ax_res, eta_stats_res, x, color='C0', label=residuals_label)
        ax_res.axhline(0, color='r', linewidth=0.7)
        ax_res.set_xlabel(xlabel)
        ax_res.set_ylabel('residuals')
        if ylimits is None:
            ax_res.set_ylim(-2e-1, 2e-1)
        else:
            ax_res.set_ylim(-ylimits[1], ylimits[1])
        if log:
            ax_res.set_yscale('log')
            
    if xlabel is not None and not plot_residuals:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    
    if ylimits is not None:
        ax.set_ylim(ylimits[0], ylimits[1])
    if xlimits is not None:
        ax.set_xlim(xlimits[0], xlimits[1])
    if log:
        ax.set_yscale('log')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    ax.legend(numpoints=3)
    if plot_residuals:
        ax_res.legend()
    fig.tight_layout()
    if save:
        plt.savefig(outfile)
        plt.close()
    else:
        plt.show()
        

def pp_plot(result, model, x, eta, power, noise, ndraws=20,
            outfile='pp_plot.png'):

    model_keys = bilby.utils.infer_parameters_from_function(model)
    model_posterior = result.posterior[model_keys]
    
    fig = plt.figure(figsize=(5,5), dpi=300)
    
    npp = 5
    ci = np.linspace(0, 1, 100)
    ci[0] = 1e-5
    
    def integrate(ind1, ind2, p, dx):
        inds = np.sort([ind1, ind2])
        return np.sum(p[inds[0]:inds[1]] * dx[inds[0]:inds[1]])

    for _ in range(npp):
        s = model_posterior.sample().to_dict('records')[0]
        model_eta = model(x, **s)
        
        sigma = np.sqrt((noise * np.exp(s['efac']))**2 + s['equad']**2)
        prob = calculate_curvature_peak_probability(power, sigma)
        
        import time
        deta = eta[0,1:] - eta[0,:-1]
        outlier_1sig = []
        outlier_2sig = []
        outlier_3sig = []
        other = []
        
        n_inside = {}
        for n in ci:
            n_inside[str(n)] = 0
            
        step = 10
        
        for i,p in enumerate(prob):
            ps = np.flip(np.sort(p))
            
            integral = np.sum(p[:-1] * deta)
            onesig = 0.683 * integral
            twosig = 0.955 * integral
            threesig = 0.997 * integral
            
            meta = 1 / np.sqrt(model_eta[i])
            meta_ind = np.argmin(np.abs((eta / 10) - meta))
            
            def calc_area(index, a, plot_area, color):
                inds = []
                inds.append(np.argwhere((p == ps[index])).flatten()[0])
                p_sub = p - ps[index]
                p_sub_pair1 = p_sub.reshape(int(len(p)/2), 2)
                p_sub_pair2 = p_sub[1:-1].reshape(int(len(p)/2)-1, 2)
                for k in range(len(p_sub_pair2)):
                    pair1 = p_sub_pair1[k]
                    pair2 = p_sub_pair2[k]
                    cond11 = (pair1[0] < 0) and (pair1[1] > 0)
                    cond12 = (pair1[1] < 0) and (pair1[0] > 0)
                    cond21 = (pair2[0] < 0) and (pair2[1] > 0)
                    cond22 = (pair2[1] < 0) and (pair2[0] > 0)
                    if cond11 or cond12:
                        inds.append(2 * k)
                    elif cond21 or cond22:
                        inds.append(2 * k)
                inds = np.sort(np.array(inds))
                
                area = 0
                if len(inds) % 2 != 0:
                    if p[0] >= ps[index]:
                        inds = np.insert(inds, 0, 0)
                    elif p[-1] >= ps[index]:
                        inds = np.append(inds, len(p)-1)
                    else:
                        rem = []
                        for k, ind in enumerate(inds):
                            if ind != 0:
                                if (p[ind - 1] <= p[ind]) and (p[ind + 1] <= p[ind]):
                                    rem.append(k)
                        rem = np.array(rem)
                        if len(rem) != 0:
                            inds = np.delete(inds, rem)
                if len(inds) % 2 == 0:
                    inds_pairs = inds.reshape(int(len(inds)/2), 2)
                    for pair in inds_pairs:
                        area += integrate(pair[0], pair[1], p, deta)

                    if plot_area:
                        plt.plot(eta[0], p)
                        for pair in inds_pairs:
                            plt.axvspan(eta[0, pair[0]], eta[0, pair[1]], color=color, alpha=0.5)
                        plt.axhline(ps[index], color='k')
                        plt.show()
                else:
                    inds_pairs = None
                    area = a

                index += step
                
                return index, area, inds_pairs
            
            j = 1
            area = 0
            
            ci_ind = 0
            inside = False
            while not inside:
                n = ci[ci_ind]
                while (area < n*integral) and (j < (len(ps)-step)):
                    j, area, inds_pairs = calc_area(j, area, False, 'red')
                
                for pair in inds_pairs:
                    if (meta_ind > pair[0]) and (meta_ind < pair[1]):
                        inside = True
                if not inside:    
                    ci_ind += 1
                if ci_ind == (len(ci)-1):
                    inside = True
            for l in range(ci_ind, len(ci)):
                n_inside[str(ci[l])] = n_inside[str(ci[l])] + 1
            
        fraction = []
        for n in ci:
            fraction.append(n_inside[str(n)])
        fraction = np.array(fraction) / len(prob)
        
        plt.plot(ci, fraction, color='C0', alpha=0.4)
        print('done {}'.format(_))
    plt.plot(ci, ci, color='r', linewidth=1.5)
    plt.xlabel('Confidence interval')
    plt.ylabel('Fraction of observations with model inside C.I.')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.grid()
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.show()
