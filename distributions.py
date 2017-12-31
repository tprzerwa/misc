from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
import scipy
import scipy.stats
from statsmodels.distributions import ECDF


class EmpiricalDistribution:
    """
    Generates empirical distribution. Mimics scipy.stats distributions behaviour.

    Usage:

    sample = np.random.binomial(2, 0.4, size=1000)
    vals, freqs = np.unique(sample, return_counts=True)
    empirical = EmpiricalDistribution(vals, freqs/np.sum(freqs))
    print(empirical.rvs(10)) # [ 1.  1.  0.  0.  0.  1.  1.  0.  1.  2.]

    """
    def __init__(self, vals, probs, loc=0., scale=1.):
        """

        :param vals: values in sample
        :param probs: frequencies of observed values
        :param loc: location parameter
        :param scale: scale parameter
        """
        self._vals = vals
        self._probs = probs
        self._loc = loc
        self._scale = scale

    def rvs(self, size=1):
        """
        Generates random sample.

        :param size: shape of sample to generate
        :return: np.ndarray
        """
        return np.random.choice(self._vals, size=size, replace=True, p=self._probs)/self._scale + self._loc


class ContinuousDistributionFitter:
    """"
    Class for fitting a continuous distributions (available in scipy.stats) into data.

    Usage:

    sample = np.random.normal(size=10000)
    fitter = ContinuousDistributionFitter()
    fitted = fitter.fit(sample, criterion='BIC', n_best=1)
    print(fitted) # [('norm', <scipy.stats._continuous_distns.norm_gen object at 0x0000000007F77C88>, (0.0095883702145322935, 0.99449724228777148), 28283.156517594962)]

    """

    def __init__(self, distributions=None):
        """
        :param distributions: (optional) iterable of scipy.stats distributions names
        """
        ### remember about von mises! this distribution doesn't work well in scipy
        scipy_continous = [a for a in dir(scipy.stats)
                           if scipy.stats._distn_infrastructure.rv_continuous in eval('scipy.stats.' + a).__class__.__bases__ if a != 'vonmises']

        self._chosen_distributions = scipy_continous if distributions is None else distributions
        self._distributions = {}

    def fit(self, x, criterion='AICc', n_best=1):
        """
        Fit scipy.stats continuous distributions to given data and choose best ones.

        :param x: (iterable) sample of unknown distribution
        :param criterion: criterion for choosing distribution; available criteria:
                            'AICc' (default) - corrected Akaike's information criterion,
                            'AIC' - Akaike's information criterion,
                            'BIC' - Bayesian information criterion,
                            'Kolmogorov' - Kolmogorov distance criterion
        :param n_best: (default=1) number of best distributions to return
        :return: list of tuples of the form
                (<scipy.stats distribution name>, <scipy.stats distribution object>, <fitted params>, <criterion value>)
                sorted in order starting from the best distribution

        Note: Usually generates numerical warnings when distribution isn't likely to fit the sample.
        """

        self._fit_distributions(x)

        if criterion == 'AIC':
            return self._fit_akaike(x, n_best)
        elif criterion == 'AICc':
            return self._fit_akaike_corrected(x, n_best)
        elif criterion == 'BIC':
            return self._fit_bic(x, n_best)
        elif criterion == 'Kolmogorov':
            return self._fit_kolmogorov(x, n_best)
        else:
            raise RuntimeError('Criterion not recognised.')

    def _fit_distributions(self, x):
        self._distributions = deepcopy({distr: {'obj': eval('scipy.stats.' + distr)} for distr in self._chosen_distributions})
        distributions_temp = {}
        np.seterr(all='raise')
        for distr_name, distr_dict in self._distributions.items():
            try:
                distr_dict['fitted_params'] = distr_dict['obj'].fit(x)
                distributions_temp[distr_name] = distr_dict
            except:
                pass
        self._distributions = distributions_temp

    def __sum_loglikelihood(self, x):
        for distr_name, distr_dict in self._distributions.items():
            try:
                distr_dict['loglikelik'] = \
                    np.sum(np.log(distr_dict['obj'].pdf(xi, *distr_dict['fitted_params'])) for xi in x)
            except:
                distr_dict['loglikelik'] = float('inf')

    def __fit_akaike(self, x, n_best):
        self.__sum_loglikelihood(x)
        for distr_name, distr_dict in self._distributions.items():
            k = len(distr_dict['fitted_params'])
            distr_dict['aic'] = 2.0*(k - distr_dict['loglikelik'])

        return sorted(self._prepare_sesults('aic'), key=itemgetter(3))[:n_best]

    def __fit_akaike_corrected(self, x, n_best):
        n = len(x)
        self.__sum_loglikelihood(x)
        for distr_name, distr_dict in self._distributions.items():
            k = len(distr_dict['fitted_params'])
            distr_dict['aic_c'] = 2.0*(k - distr_dict['loglikelik'] + k*(k+1)/(n-k-1))

        return sorted(self._prepare_sesults('aic_c'), key=itemgetter(3))[:n_best]

    def __fit_bic(self, x, n_best):
        n = len(x)
        self._sum_loglikelihood(x)
        for distr_name, distr_dict in self.__distributions.items():
            k = len(distr_dict['fitted_params'])
            distr_dict['bic'] = -2.0*distr_dict['loglikelik'] + k*(np.log(n) - np.log(2.0*np.pi))

        return sorted(self._prepare_sesults('loglikelik'), key=itemgetter(3))[:n_best]

    def _fit_kolmogorov(self, x, n_best):
        for distr_name, distr_dict in self._distributions.items():
            ecdf = ECDF(x)
            distr_dict['kolmogorov'] = np.max(np.abs(ecdf(x) - distr_dict['obj'].cdf(x, *distr_dict['fitted_params'])))

        return sorted(self._prepare_sesults('kolmogorov'), key=itemgetter(3))[:n_best]

    def _prepare_sesults(self, criterion_key):
        return [(distr, distr_dict['obj'], distr_dict['fitted_params'], distr_dict[criterion_key])
                for distr, distr_dict in self._distributions.items() if np.isfinite(distr_dict[criterion_key])]



# class for clear code, but truly works as function
class plot_fit:
    """
    Helper class for visualising accuracy of results obtained by ContinuousDistributionFitter.

    Usage:

    sample = np.random.normal(size=10000)
    fitter = ContinuousDistributionFitter()
    results = fitter.fit(sample, criterion='BIC', n_best=3)
    plot_fit(sample, results) # generates chart


    Note: Tested only in 1080p resolution.
    """
    def __init__(self, data, distributions, bins=None, title=None):
        """
        Visualises accuracy of results obtained by ContinuousDistributionFitter.

        :param data: (iterable) empirical sample
        :param distributions: output from ContinuousDistributtionFitter.fit method
        :param bins: (optional) number of bins for histogram
        :param title: (optional) title for figure
        """
        self._data = data
        self._distributions = distributions
        self._bins = bins
        self._title = title

        self._data_rng, self._linspace = self._get_ranges()
        self._number_of_distributions = len(self._distributions)

        self._fig = plt.figure(1)
        self._colormap, self._line_styles = self._set_up_figure()

        self._pdf_plot = plt.subplot(121)
        self._make_pdf_plot()
        self._cdf_plot = plt.subplot(122)
        self._make_cdf_plot()

        plt.show()

    def _set_up_figure(self):
        colormap = plt.cm.Set1
        plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, self._number_of_distributions)])
        line_styles = ['-', '--'] * (self._number_of_distributions // 2 + 1)
        self._fig.tight_layout()
        return colormap, line_styles

    def _make_pdf_plot(self):
        if self._title is not None:
            self._pdf_plot.set_title('PDF: ' + self._title)

        s = np.std(self._data)
        if self.__bins is None:
            self.__bins = np.int32(np.ceil((self._data_rng*len(self._data)**(1.0/3.0))/(3.49*s)))

        heights, _, _ = self._pdf_plot.hist(self._data, label='data', normed=True, bins=self.__bins, color='lightgreen')
        if self._title is not None:
            print(self._title)

        for i, distr in enumerate(self._distributions):
            print(i+1, distr)
            self._pdf_plot.plot(self._linspace, distr[1].pdf(self._linspace, *distr[2]), label=distr[0],
                                 linestyle=self._line_styles[i])

        self._pdf_plot.set_ylim((0, 1.2*max(heights)))

        self._pdf_plot.legend()

    def _make_cdf_plot(self):
        if self._title is not None:
            self._cdf_plot.set_title('CDF: ' + self._title)

        ecdf = ECDF(self._data)

        for i, distr in enumerate(self._distributions):
            self._cdf_plot.plot(self._linspace, distr[1].cdf(self._linspace, *distr[2]), label=distr[0],
                                 linestyle=self._line_styles[i])

        self._cdf_plot.plot(self._linspace, ecdf(self._linspace), label='empirical', color='k')
        self._cdf_plot.set_ylim((-0.02, 1.02))
        self._cdf_plot.legend(loc=4)

    def __get_ranges(self):
        min_data, max_data = np.min(self._data), np.max(self._data)
        rng = max_data - min_data
        l_end, r_end = min_data - 0.1*rng, max_data + 0.1*rng
        return rng, np.linspace(l_end, r_end, 500)
