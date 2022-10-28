import numpy as np
import scipy.special, scipy.stats as st
from astropy.table import Table
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
import emcee, corner
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def gauss(x, mu, sig):
    """
    Return the Gaussian distribution for a given mean and standard deviation

    Parameters:
    -----------------------------------------
    x: int or list
        The random variable
    mu: int
        The mean of the gaussian
    sig: int
        standard deviation of the gaussian
    -----------------------------------------
    """
    norm = 1 / (sig * np.sqrt(2 * np.pi))
    dis = norm * np.exp(-((x - mu) ** 2) / (2 * sig**2))
    return dis


def gauss_mix(x, mus, sigmas, weights):
    """
    Return the Gaussian mixture distribution for n components

    Parameters:
    -----------------------------------------
    x: int or list
        The random variable
    mus: list
        The means of the individual components (length n)
    sigmas: list
        The standard deviations of the individual components (length n)
    weights: list
        The individual weights of the components (length n), need to add up to 1
    -----------------------------------------
    """
    mixture = np.zeros_like(x)

    # Check f weights add up to 1
    norm = np.sum(weights)
    if abs(norm - 1) > 1e-5:
        print("Error !!! The weights don't add up to 1. Returning empty array")
        return mixture
    for comp in range(len(mus)):
        ind_comp = gauss(x, mus[comp], sigmas[comp])
        mixture += weights[comp] * ind_comp
    return mixture


def exp_mod_gauss(x, mu, sig, tau):
    """
    Return the exponentially modified Gaussian distribution.

    See https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution (equation 1)

    Parameters:
    -----------------------------------------
    x: int
        The random variable
    mu: int
        The mean of the Gaussian
    sig: int
        The standard deviation of the Gaussina
    tau: int
        The decay constant/relaxation time of the exponential
    -----------------------------------------
    """
    gauss_arg = (mu + (sig**2 / tau) - x) / (sig * np.sqrt(2))
    errc_fun = scipy.special.erfc(gauss_arg)
    if np.max(((1 / 2.0 / tau**2) * (2 * mu * tau + sig**2 - 2 * x * tau))) > 100:
        print(
            mu,
            sig,
            tau,
            np.max((1 / 2.0 / tau**2) * (2 * mu * tau + sig**2 - 2 * x * tau)),
        )
    exp_term = np.exp((1 / 2.0 / tau**2) * (2 * mu * tau + sig**2 - 2 * x * tau))
    dis = (1 / 2.0 / tau) * exp_term * errc_fun
    return dis


def fit_func(x, *params):
    """
    Return the two component ON distribution in the case of exponentially modified gaussian

    This is used to get an initial fit for the ON histogram. Return the mixtuure PDF

    Parameters:
    -----------------------------------------
    x: list
        The random variable
    params: list
        The parameters describing the ON distribution.

        2 if it is just the OFF component [mu, sig]
        6 if it is ON component: 2 for OFF, 3 for ON and NF
        [mu_off, mu_on, sif_off, sig_on, tau, nf]
    -----------------------------------------
    """
    if len(params) == 2:
        return gauss(x, params[0], params[1])
    elif len(params) == 6:
        mu0, mu1, sig0, sig1, tau, c0 = params
        off_comp = c0 * gauss(x, mu0, sig0)
        on_comp = (1 - c0) * exp_mod_gauss(x, mu1, sig1, tau)
        return off_comp + on_comp


class nulls:
    """
    The main class object that does the fitting for the ON/OFF histograms

    Parameters:
    -------------------------------------------
    on: list
        The ON histogram
    off: list
        The OFF histogram
    ncomp: int
        The number of components for the mixture model. Can be anything for the
        Gaussian mixture model. Only supporeted for 2 components in the case of
        exponentially modified gaussian case (EMG) (1 for OFF, 1 for ON)
    nwalkers: int
        The number of walkers used in the MCMC fit, default=32
    model: str
        The mixture model to be used for ON histogram
        'gauss' for GMM, 'exp_tail' for EMG
    burnin: int
        The number of steps used for burnin for the MCMC run, default=200
    nsteps: int
        The length of the chain (the number of steps, MCMC is run for), default=5000


    Sample usage:
    -------------------------------------------
    null = nulls(on, off, 2, model=model, burnin=200, nsteps=5000)
    null.run_mcmc(corr=False)
    null.plot_fit()
    null.gen_kde()
    null.ritchings_fit()
    """

    def __init__(
        self, on, off, ncomp, nwalkers=32, model="gauss", burnin=200, nsteps=5000
    ):

        # Get rid of zapped single pulses
        mask = ~np.isnan(on) & ~np.isnan(off)
        on = on[mask]
        off = off[mask]
        self.on = on
        self.off = off

        # Remove outliers
        # mask = np.abs(on-off)>15
        # self.on = on[~mask]
        # self.off = off[~mask]

        self.n = ncomp
        self.nwalkers = nwalkers
        self.model = model
        self.burn = burnin
        self.mcmc_steps = nsteps

        self.bins = np.linspace(
            min(np.min(self.on), np.min(self.off)),
            max(np.max(self.on), np.max(self.off)),
            1000,
        )

        return None

    def gen_kde(self):
        """
        Funtion to generate the KDEs for on and off histograms

        The kernel is Gaussian an the initial bandwidth is chosen
        using Silverman's rule of thumb where bw = ((d+1)*n/4)**(-1/(d+4))
        where n is the number of data points and d is the dimensionality
        """
        n, d = len(self.off), 1
        bw = ((d + 2) * n / 4) ** (-1 / (d + 4))
        offkde = KernelDensity(bandwidth=bw, kernel="gaussian")
        onkde = KernelDensity(bandwidth=bw, kernel="gaussian")
        offkde.fit(self.off[:, np.newaxis])
        onkde.fit(self.on[:, np.newaxis])
        self.offkde = offkde
        self.onkde = onkde
        return None

    def ritchings_fit(self, plot=False, save_txt=False):
        """
        Funtion to emulate ritchings (1976) nulling estimate

        Generates realizations using the KDEs, and estimates NF
        """
        bins = self.bins
        mask = np.where(bins <= 0, 1, 0)
        off_realz = np.exp(self.offkde.score_samples(bins[:, np.newaxis]))
        on_realz = np.exp(self.onkde.score_samples(bins[:, np.newaxis]))
        nf = np.linspace(0, 1, 10000)
        res = on_realz * mask - nf[:, np.newaxis] * (off_realz * mask)

        chi2 = np.abs(np.sum(res, 1))
        best_fit_nf = nf[np.argmin(chi2)]
        self.ritchings_nf = best_fit_nf
        self.ritchings_chi2 = np.array([nf, chi2])

        if plot:
            plt.clf()
            plt.hist(self.off, 50, density=True, color="gray", alpha=0.5)
            plt.hist(self.on, 100, density=True, color="gray", alpha=0.5)
            plt.fill_between(bins, off_realz, alpha=0.5, label="OFF")
            plt.fill_between(bins, on_realz, alpha=0.5, label="ON")
            plt.fill_between(bins, best_fit_nf * off_realz, alpha=0.5, label="NF*OFF")
            plt.fill_between(
                bins, on_realz - best_fit_nf * off_realz, alpha=0.5, label="ON-NF*OFF"
            )
            plt.legend()
            plt.xlabel("Raw Intensity", fontsize=15)
            plt.ylabel("PDF", fontsize=15)
            plt.title(f"Ritchings NF estimate = {np.round(best_fit_nf*100, 2)}%")
            plt.tight_layout()
            plt.show()

        if save_txt:
            tab = Table([bins, off_realz, on_realz], names=["bins", "off", "on"])
            meta = {}
            meta["nf"] = best_fit_nf
            meta["chi2"] = chi2
            tab.meta = meta
            tab.write(
                f"{self.args.basename}/nf_ritchings.txt",
                format="ascii.ecsv",
                overwrite=True,
            )

        return None

    def get_off_hist_guess(self):
        """
        Get the fit for OFF histogram using EM algorithm in scikit-learn

        Estimates the OFF histogram by a gaussian.
        """
        off = self.off
        gm = GaussianMixture(n_components=1, tol=1e-4)
        gm.fit(off[:, np.newaxis])
        mean, std = gm.means_[0][0], np.sqrt(gm.covariances_[0][0][0])

        self.off_mean = mean
        self.off_std = std
        self.off_mean_err = std / np.sqrt(len(off))
        self.off_std_err = std * np.sqrt(2 / (len(off) - 1))
        return None

    def get_on_hist(self, nbins=1000):
        """
        Function to get the histogram of the on window.

        Especially useful when getting an initial fit for the ON histigrams
        using EMG
        """
        on = self.on
        hist, bins = np.histogram(on, nbins, density=True)
        bins = 0.5 * (bins[1:] + bins[:-1])
        self.on_hist, self.bins = hist, bins
        return None

    def get_on_hist_guess(self):
        """
        Get the initial fit for ON histogram

        Estimates the ON histogram by a gaussian using EM algorithm in scikit-learn if gaussian
        else does a least squares fit if it is an EMG.
        """
        on = self.on
        off = self.off
        model = self.model
        ncomp = self.n

        # Fit using GMM
        gm = GaussianMixture(n_components=ncomp, tol=1e-4)
        gm.fit(on[:, np.newaxis])
        sort = np.argsort(np.abs(gm.means_.flatten()) - self.off_mean)
        mean, std = gm.means_.flatten()[sort], np.sqrt(gm.covariances_.flatten())[sort]
        weights = gm.weights_.flatten()[sort]

        # Store them
        self.on_means = mean
        self.on_std = std
        self.weights = weights
        init_params = np.concatenate((mean, std, weights[:-1]))

        if model == "gauss":
            init_on_params = init_params
            init_on_param_err = None
            # print(init_on_params)
        elif model == "exp_tail":
            tau_guess = 1
            init_on_comp_params = np.insert(init_params, -1, tau_guess)

            self.get_on_hist()

            bounds_lower = [np.min(off), np.min(on), 0, 0, 0, 0]
            bounds_upper = [
                np.max(off),
                np.max(on),
                np.max(off),
                np.max(on),
                np.max(on),
                1,
            ]

            bounds = (bounds_lower, bounds_upper)
            # print(init_on_comp_params)
            # print(bounds)

            on_fit = init_on_comp_params
            print(on_fit)
            on_conv = np.ones((len(init_on_comp_params), len(init_on_comp_params))) * 50
            max_tries = 5
            curr_try = 0
            while np.any(np.abs(on_conv) > 10):
                curr_try += 1
                on_fit, on_conv = curve_fit(
                    fit_func,
                    self.bins,
                    self.on_hist,
                    p0=on_fit,
                    bounds=bounds,
                    method="trf",
                    maxfev=5000,
                )

                if curr_try >= max_tries:
                    print(
                        """Failed to obtain a good fit for ON histogram after 25000 steps.
                    Try changing the intial parameters for the least square fit"""
                    )
                    break

            init_on_params = on_fit
            init_on_param_err = np.diagonal(on_conv) ** 0.5
            # print(init_on_params, init_on_param_err, np.max(self.on))

        self.on_init_guess = init_on_params
        self.on_err_init_guess = init_on_param_err
        return None

    def get_p0_mcmc(self):
        """
        Initialize the walkers around 3-sigma region of the initial fit

        For a few parameters which are not allowed to be <0, use truncated normal.
        """

        n = self.n
        nwalk = self.nwalkers

        if self.on_err_init_guess is None:
            # Means the fit is using GMM

            # For the means, use the means and std using the on fit
            rand_means = np.random.normal(
                loc=self.on_means, scale=self.on_std, size=(nwalk, n)
            )

            # For stds, make sure the guess doesn't go below 0
            rand_stds = st.truncnorm.rvs(
                0, np.max(self.on), loc=self.on_std, scale=self.on_std, size=(nwalk, n)
            )

            # For weights use dirichilet distribution
            rand_weights = st.dirichlet.rvs(alpha=np.ones(n), size=(nwalk,))

            # Stack them up
            pstart = np.hstack((rand_means, rand_stds, rand_weights[:, :-1]))

        else:
            # Means its a gaussina fit with exponential tail
            fit = self.on_init_guess
            err = self.on_err_init_guess
            # print(fit, err)
            rand_means = np.random.normal(loc=fit[:n], scale=err[:n], size=(nwalk, n))
            rand_stds = st.truncnorm.rvs(
                0,
                np.max(self.on),
                loc=fit[n : 2 * n],
                scale=err[n : 2 * n],
                size=(nwalk, n),
            )
            rand_taus = st.truncnorm.rvs(
                0.2,
                np.max(self.on),
                loc=fit[2 * n : 3 * n - 1],
                scale=0.5 * err[2 * n : 3 * n - 1],
                size=(nwalk, n - 1),
            )
            # overflow = rand_lams>=10
            # if len(overflow)!=0:
            #     rand_lams[overflow] = np.random.random(rand_lams[overflow].shape[0])*10
            rand_weights = st.dirichlet.rvs(alpha=np.ones(n), size=(nwalk,))

            # Stack them up
            pstart = np.hstack((rand_means, rand_stds, rand_taus, rand_weights[:, :-1]))

        self.pstart = pstart
        # print(pstart, pstart.shape)
        return None

    def log_likelihood(self, params):
        """
        The log-likelihood function, total likelihood is product of individual likelihoods

        And the individual likelihood is estimated from the PDF

        Parameters:
        -----------------------------------------
        x: list
            The ON intensities
        params: list
            The parameters of the ON distribution, differs for GMM, EMG
        -----------------------------------------
        """
        x = self.on
        n = self.n
        mus = params[:n]
        sigmas = params[n : 2 * n]
        if self.model == "gauss":
            weights = np.zeros_like(mus)
            weights[:-1] = params[2 * n :]
            weights[-1] = 1 - weights[:-1].sum()
            prob = gauss_mix(x, mus=mus, sigmas=sigmas, weights=weights)

        elif self.model == "exp_tail":
            tau = params[2 * n : 3 * n - 1]
            weights = np.zeros_like(mus)
            weights[:-1] = params[3 * n - 1 :]
            weights[-1] = 1 - weights[:-1].sum()
            if n == 1:
                prob = exp_mod_gauss(x, mus[0], sigmas[0], tau) * weights[0]
            else:
                prob1 = gauss(x, mus[0], sigmas[0]) * weights[0]
                prob2 = exp_mod_gauss(x, mus[1], sigmas[1], tau) * weights[1]
                prob = prob1 + prob2

        likelihood = prob
        likelihood[likelihood == 0] = 1e-50
        self.likelival = np.log(likelihood).sum()
        return None

    def log_prior(self, params):
        """
        The log-prior function that sets the priors on the individual parameters

        The priors for the OFF parameters are gaussians around the initial OFF fit,
        those for ON distributions are uniform and those of NF follow
        dirichilet distribution. The value of tau can't be arbitrarily close to 0.
        That would cause overflow in the exponent and also given extremely steep
        distributions. So we cut it off at 0.2

        Parameters:
        -----------------------------------------
        params: list
            The parameters of the ON distribution, differs for GMM, EMG
        -----------------------------------------
        """
        n = self.n
        on = self.on
        mus = params[:n]
        sigmas = params[n : 2 * n]
        if self.model == "gauss":
            weights = np.zeros_like(mus)
            weights[:-1] = params[2 * n :]
            weights[-1] = 1 - weights[:-1].sum()
        elif self.model == "exp_tail":
            tau = params[2 * n : 3 * n - 1]
            weights = np.zeros_like(mus)
            weights[:-1] = params[3 * n - 1 :]
            weights[-1] = 1 - weights[:-1].sum()

        # Put prior on mus
        if np.any(mus < np.min(self.off)):
            mu_sum = -np.inf
        elif np.any(mus > np.max(self.on)):
            mu_sum = -np.inf
        else:
            # Now comes the off component prior
            mu_sum = -0.5 * (mus[0] - self.off_mean) ** 2 / (2 * self.off_mean_err**2)

        # Prior mask on stds
        if np.any(sigmas < 0):
            sigma_sum = -np.inf
        elif np.any(sigmas > np.max(self.on)):
            sigma_sum = -np.inf
        else:
            sigma_sum = (
                -0.5 * (sigmas[0] - self.off_std) ** 2 / (2 * self.off_std_err**2)
            )

        # Put mask on weights
        if np.any(weights < 0):
            weights_sum = -np.inf
        else:
            weights_sum = 0

        # May be om lams
        if self.model == "exp_tail":
            if np.any(tau < 0.2):
                weights_sum += -np.inf
            elif np.any(tau > np.max(on)):
                weights_sum += -np.inf
            else:
                weights_sum += 0

        self.priorval = mu_sum + sigma_sum + weights_sum
        return None

    def log_post(self, params):
        """
        The log-posterior function

        Parameters:
        -----------------------------------------
        params: list
            The parameters of the ON distribution, differs for GMM, EMG
        -----------------------------------------
        """
        self.log_prior(params)
        if np.isinf(self.priorval):
            post = -np.inf
        else:
            self.log_likelihood(params)
            post = self.likelival + self.priorval
        return post

    def run_mcmc(self, corr=True):
        """
        Function to runs the MCMC.

        Takes the walkers, their initial positions and explores the parameter space

        Parameters:
        -----------------------------------------
        corr: bool
            Flag that decides whether to take correlation length into account.
        -----------------------------------------
        """
        nwal = self.nwalkers
        self.get_off_hist_guess()
        self.get_on_hist_guess()
        ndim = len(self.on_init_guess)
        sampler = emcee.EnsembleSampler(
            nwalkers=nwal, ndim=ndim, log_prob_fn=self.log_post
        )
        self.get_p0_mcmc()
        state = sampler.run_mcmc(self.pstart, self.burn, progress=True)
        sampler.reset()
        sampler.run_mcmc(state, self.mcmc_steps, progress=True)
        samples = sampler.get_chain(flat=True)
        self.all_samples = samples
        self.sampler = sampler

        if corr:
            # Account for  autocorrelation length
            autocorrbool = True
            autocorriter = 0
            while autocorrbool:
                corr_len = int(np.max(sampler.get_autocorr_time(quiet=True)))
                if corr_len > self.mcmc_steps / 50:
                    print(
                        f"\n Not enough independent samples: given the chain length of {self.mcmc_steps}\
                        and the max correlation length of {corr_len}, so run ~{corr_len-self.mcmc_steps/50} more steps"
                    )
                    steps = int(2 * (corr_len - self.mcmc_steps / 50) * 50)
                    sampler.run_mcmc(initial_state=None, nsteps=steps, progress=True)
                    autocorriter += 1
                    self.mcmc_steps += steps
                else:
                    autocorrbool = False

            if autocorriter >= 10:
                print(
                    "Not enough samples even after 10 iterations (5000 runs).\
                    Please verify the initial conditions and nwalkers, acceptance function"
                )
                raise SystemExit

            ind = [corr_len * i for i in range(int(len(samples) / corr_len))]
            mask = np.zeros(len(samples))
            mask[ind] = 1
            mask = mask.astype(bool)
            self.samples = samples[mask]
        else:
            self.samples = samples

        # Estimate AIC and BIC values
        self.AIC()
        self.BIC()

        return None

    def AIC(self):
        """
        Funtion to estimate AIC
        """
        samples = self.samples
        fit_val = np.median(samples, axis=0)
        self.log_likelihood(fit_val)  # Maximum value of log likelihood
        k = len(fit_val)  # Number of parameters to be fit
        self.aic_val = 2 * k - 2 * self.likelival
        return None

    def BIC(self):
        """
        Funtion to estimate BIC
        """
        samples = self.samples
        fit_val = np.median(samples, axis=0)
        self.log_likelihood(fit_val)  # Maximum value of log likelihood
        k = len(fit_val)  # Number of parameters to be fit
        self.bic_val = k * np.log(len(self.on)) - 2 * self.likelival
        return None

    def plot_fit(self):
        """
        Function to plot the best fit MCMC values

        Generates the corner plot, the ON/OFF histograms with best fit analytical models.
        """
        fig = plt.figure()

        off = self.off
        on = self.on
        n = self.n
        samples = self.samples
        model = self.model

        if self.model == "gauss":
            labels = []
            for i in range(self.n):
                labels.append([rf"$\mu_{i}$", rf"$\sigma_{i}$", rf"$c_{i}$"])
            labels[0][2] = "NF"
            labels = np.array(labels)
            labels = labels.flatten("F")
            # labels = np.concatenate((labels, ['NF']))
            labels = labels[:-1]
        elif self.model == "exp_tail":
            labels = []
            lam_labels = []
            for i in range(self.n):
                labels.append([rf"$\mu_{i}$", rf"$\sigma_{i}$"])
                if i > 0:
                    lam_labels.append(rf"$\tau_{i}$")
            labels = np.array(labels)
            labels = labels.flatten("F")
            labels = np.concatenate((labels, lam_labels, ["NF"]))
        corner.corner(samples, labels=labels, label_kwargs={"fontsize": 15})
        plt.show()

        fit_values = np.median(samples, axis=0)

        # Plot histograms
        fig2 = plt.figure()
        _ = plt.hist(off, 50, density=True, histtype="step", label="Off hist.")
        _ = plt.hist(on, 75, density=True, histtype="step", label="On hist.")
        x_plot = self.bins

        # Plot fit components
        off_fit = gauss(x_plot, fit_values[0], fit_values[n])
        plt.plot(x_plot, off_fit, label="Off fit")

        # Iterate for on-fit
        mus = fit_values[1:n]
        stds = fit_values[n + 1 : 2 * n]
        if model == "gauss":
            weights = fit_values[2 * n :]
            weights = np.concatenate((weights, [1 - weights.sum()]))
            plt.plot(x_plot, weights[0] * off_fit, label="Off fit*NF")

            on_fit = off_fit * weights[0]
            for i in range(len(mus)):
                on_fit_comp = gauss(x_plot, mus[i], stds[i]) * weights[i + 1]
                plt.plot(
                    x_plot, on_fit_comp * (1 - fit_values[-1]), label=f"On comp. {i+2}"
                )
                on_fit += on_fit_comp
        elif model == "exp_tail":
            lams = fit_values[2 * n : 3 * n - 1]
            weights = fit_values[3 * n - 1 :]
            weights = np.concatenate((weights, [1 - weights.sum()]))
            plt.plot(x_plot, weights[0] * off_fit, label="Off fit*NF")

            on_fit_off = off_fit * weights[0]
            on_fit = np.zeros_like(x_plot)
            for i in range(len(mus)):
                on_fit_comp = (
                    exp_mod_gauss(x_plot, mus[i], stds[i], lams[i]) * weights[i + 1]
                )
                on_fit += on_fit_comp
            plt.plot(x_plot, on_fit, label=f"On comp {i+2}")
            on_fit += on_fit_off

        plt.plot(x_plot, on_fit, label="On fit")
        plt.xlabel("Raw Intensity", fontsize=15)
        plt.ylabel("Probability Density", fontsize=15)
        plt.legend()
        plt.tight_layout()
        plt.show()
