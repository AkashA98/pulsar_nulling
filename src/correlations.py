import numpy as np
from astropy.table import Table
from tqdm import tqdm
from gen_nf import gauss, gauss_mix, exp_mod_gauss
from sklearn.neighbors import KernelDensity
from scipy.optimize import curve_fit
from emcee import EnsembleSampler
from scipy.stats import chi2
from matplotlib import pyplot as plt


class emcee_fit:
    """
    MCMC algorithm for fitting null and burst length histograms

    Parameters:
    -------------------------------
    hist: list
        List of null/burst lengths
    init_sol: list
        Initial guess solution for the decay constant parameter
    nwalkers: int
        The number of walkers used in the MCMC fit, default=32
    ndim: int
        Number of parameter in the fit, default=1, the decay constant
    burn: int
        The number of steps used for burnin for the MCMC run, default=200
    steps: int
        The length of the chain (the number of steps, MCMC is run for), default=5000
    """

    def __init__(self, hist, init_sol, nwalk=32, ndim=1, burn=200, steps=2000):
        self.hist = hist
        self.init_sol = init_sol
        self.nwalkers = nwalk
        self.ndim = ndim
        self.burn = burn
        self.steps = steps
        return None

    def func(self, x, a):
        """
        The analytical function that describes the exponential PDF.

        Slightly modified because the peak will be at 1

        Parameters:
        ------------------------------------
        x: list
            The null/burst lengths
        a: float
            The decay constant
        ------------------------------------
        """
        return a * np.exp(-(x - 1) * a)

    def loglike(self, a):
        """
        The log-likelihood function

        Parameters:
        ------------------------------------
        a: float
            The decay constant
        ------------------------------------
        """
        like = self.func(self.hist, a)
        like[like == 0] = 1e-50
        if np.any(like <= 0):
            print("Error")
            print(a)
        logl = np.sum(np.log(like))
        return logl

    def logp(self, a):
        """
        The log-prior function

        the value of decay consatnt can't be less than 0

        Parameters:
        ------------------------------------
        a: float
            The decay constant
        ------------------------------------
        """
        if a <= 0:
            return -np.inf
        elif a >= np.max(self.hist):
            return -np.inf
        else:
            return 0

    def logpost(self, a):
        """
        The log-posterior function

        Parameters:
        ------------------------------------
        a: float
            The decay constant
        ------------------------------------
        """
        logpr = self.logp(a)
        if np.isinf(logpr):
            return -np.inf
        else:
            logl = self.loglike(a)
            logpo = logl + logpr
            return logpo

    def run(self):
        """
        Function that runs the MCMC chain
        """
        sampler = EnsembleSampler(nwalkers=32, ndim=1, log_prob_fn=self.logpost)
        state = sampler.run_mcmc(
            1e-3 * np.random.rand(32, 1) + self.init_sol, self.burn, progress=True
        )
        sampler.run_mcmc(state, self.steps, progress=True)
        samples = sampler.get_chain(flat=True)
        sol = np.median(samples)
        self.best_fit_value = sol
        self.samples = samples
        return sol


def find_runs(x):
    """
    Find runs of consecutive items in an array.

    Returns the consecutive runs, the index at which they started, the lengths

    Parameters:
    ---------------------------------------
    x: list
        The list of items for which runs have to be estimates
    """

    # Make sure this is an array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError("only 1D array supported")
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths


def model_params(txt_file):
    """
    Function that gets the model parameters from the MCMC run on the
    ON/OFF histograms. The chains have to stored for this to be saved

    Parameters:
    ------------------------------------
    txt_file: str
        The file in which the chains and the best fit values are stored
    ------------------------------------
    """
    tab = Table.read(txt_file, format="ascii")
    chains = np.zeros((len(tab), len(tab.colnames)))
    for i in range(chains.shape[-1]):
        chains[:, i] = tab[tab.colnames[i]].data
    meta = tab.meta
    ndim, fit_val, model = meta["n"], meta["fit_val"], meta["model"]
    return ndim, fit_val, model, chains


def get_nf_prob(on_intensities, model, ndim, fit_val):
    """
    Function to get the null probabilities of the single pulses

    Need the ON intensities and the model that describes the ON histrogram

    Parameters:
    ------------------------------------
    on_intensities: list
        The ON intensities
    model: str
        The model used to describe the ON histogram. Can be 'gauss' which is
        a GMM or 'exp_tail' which is exponentially modified gaussian (EMG)
    ndim: int
        The number of components used to describe the ON histograms
        This decides the model parameters which will be 3n-1 for GMM and
        4n-1 for EMG
    fit_val: list
        The fit parameters that describe the ON/OFF histograms. These are
        [means, stds, (taus), weights] where weights will be n-1. Since they add
        to 1.
    """

    if model == "gauss":
        means = fit_val[:ndim]
        sigmas = fit_val[ndim : 2 * ndim]
        weights = fit_val[2 * ndim :]
        weights = np.insert(weights, len(weights), 1 - weights.sum())
        on_prob = gauss_mix(on_intensities, mus=means, sigmas=sigmas, weights=weights)
        off_comp = gauss(on_intensities, means[0], sigmas[0]) * weights[0]
        nf_prob = off_comp / on_prob
    elif model == "exp_tail":
        means = fit_val[:ndim]
        sigmas = fit_val[ndim : 2 * ndim]
        taus = fit_val[2 * ndim : 3 * ndim - 1]
        weights = fit_val[3 * ndim - 1 :]
        weights = np.insert(weights, len(weights), 1 - weights.sum())
        off_comp = gauss(on_intensities, means[0], sigmas[0]) * weights[0]
        on_comp = np.zeros(len(on_intensities))
        for i in range(ndim - 1):
            on_comp += (
                exp_mod_gauss(on_intensities, means[i + 1], sigmas[i + 1], taus[i])
                * weights[i + 1]
            )
        on_prob = off_comp + on_comp
        nf_prob = off_comp / on_prob
    return nf_prob


def get_null_em_dist(nf_prob, fit_type="kde"):
    """
    Function that gets the null and burst length histograms and fits them

    Two types of fits are posssible:
    1. KDE (estimates the histograms with KDEs and fits the KDEs)
        Returns the best fit value
    2. MCMC
        Retruns the sampler (emcee.EnsembleSampler)

    Parameters:
    --------------------------------------
    nf_prob: list
        The null probabilities of all the single pulses
    fit_type: str
        The type of fit to be performed. Default is 'kde'
        Possible values are 'kde', 'mcmc', 'both'
    --------------------------------------
    """

    mask = np.where(nf_prob >= 0.5, 1, 0)
    val, _, run = find_runs(mask)
    null_lens = run[val.astype(bool)]
    em_lens = run[~(val.astype(bool))]

    if len(null_lens) != 0:
        null_fit = fit_hist(null_lens, fit_type)
    else:
        null_fit = np.nan
    em_fit = fit_hist(em_lens, fit_type)

    return null_lens, em_lens, null_fit, em_fit


def fit_func(x, a):
    """
    The analytical function that describes the exponential PDF.

    Slightly modified because the peak will be at 1

    Parameters:
    ------------------------------------
    x: list
        The null/burst lengths
    a: float
        The decay constant
    ------------------------------------
    """
    return a * np.exp(-(x - 1) * a)


def fit_hist(dist, fit_type="kde", bw=None, plot=False):
    """
    Function that fits the null and emission histograms.

    Two types of fits are posssible:
    1. KDE (estimates the histograms with KDEs and fits the KDEs)
    2. MCMC

    Returns the null/emission length distributions and the best fit value
    for the decay constant.

    Parameters:
    ------------------------------------
    dist: list
        The null/burst distributions
    fit_type: str
        The type of fit to be performed. Default is 'kde'
        Possible values are 'kde', 'mcmc', 'both'
    ------------------------------------
    """

    if fit_type == "kde" or fit_type == "both":
        if bw is None:
            bw = (3 * (len(dist) / 4)) ** (-0.2)

        kde1 = KernelDensity(bandwidth=bw, kernel="gaussian")
        kde2 = KernelDensity(bandwidth=bw, kernel="gaussian")
        kde1.fit(dist[:, np.newaxis])
        kde2.fit(1 - dist[:, np.newaxis])
        x = np.linspace(1, np.max(dist), 1000)
        sam = np.exp(kde1.score_samples(x[:, np.newaxis])) + np.exp(
            kde2.score_samples(x[:, np.newaxis])
        )
        p0 = [np.max(sam)]  # [1/x[np.argmin(np.abs(sam-np.max(sam)/2))]]
        popt, _ = curve_fit(fit_func, x, sam, p0=p0)

    if fit_type == "mcmc" or fit_type == "both":
        h = np.histogram(dist, 20, density=True)
        mcmc_fit = emcee_fit(dist, np.max(h[0]))
        mcmc_sol = mcmc_fit.run()
        print(mcmc_sol)

    if plot:
        if fit_type == "kde" or fit_type == "both":
            plt.plot(
                x,
                sam,
                label=f"kde fit FWHM = {x[np.argmin(np.abs(sam-np.max(sam)/2))]}",
            )
            plt.plot(
                x,
                fit_func(x, *popt),
                label=rf"exp fit: $\tau$={np.round(1/popt[0], 3)}",
            )
        if fit_type == "mcmc" or fit_type == "both":
            plt.plot(
                x,
                fit_func(x, mcmc_sol),
                label=rf"mcmc fit: $\tau$={np.round(1/mcmc_sol, 3)}",
            )
        plt.legend()
        plt.xlabel("Periods")
        plt.ylabel("PDF")
        plt.tight_layout()
        plt.show()

    if fit_type == "kde":
        return [popt]
    elif fit_type == "mcmc":

        return [mcmc_sol]
    elif fit_type == "both":
        return [popt, mcmc_sol]


def get_nf_prob_dist(nf_prob):
    """
    Function to get the distribution of null probabilities

    Helper function for estimating the significance of the power

    Parameters:
    -----------------------------------------
    nf_prob: array
        The null probabilities of single pulses
    -----------------------------------------
    """
    # Check out for any nans
    nf_prob[np.isnan(nf_prob)] = np.nanmedian(nf_prob)
    dist, bins = np.histogram(nf_prob, bins=100, density=True)
    bins = 0.5 * (bins[1:] + bins[:-1])
    return dist, bins


def plot_profile(sps):
    """
    Funtion to plot the single pulses

    Helper function for drifting analysis

    Parameters:
    -----------------------------------------
    sps: array
        The stack of single pulses, 2D array
    -----------------------------------------
    """
    plt.imshow(sps, aspect="auto", origin="lower")
    plt.xlabel("Phase bins")
    plt.ylabel("Pulse number")
    plt.title("Pulse stack")
    plt.show()


def get_on_window(sps):

    """
    Funtion to get the limits of the on-window

    Helper function for drifting analysis, returns the
    start and end of the ON window

    Parameters:
    -----------------------------------------
    sps: array
        The stack of single pulses, 2D array
    -----------------------------------------
    """

    plot_profile(sps[:, ons:one])

    # Ask the user for on-off pulse phases
    print("Please enter the pulse phases for on and off windows\n")

    onst = input("Start of on-window (Press P for the plot):")
    if ons == "P":
        plot_profile(sps[:, ons:one], extent=extent)
        ons = input("Start of on-window:")

    onen = input("End of on-window (Press P for the plot):")
    if one == "P":
        plot_profile(sps[:, ons:one], extent=extent)
        one = input("End of on-window:")

    ons, one = int(onst), int(onen)

    extent = [0, one - ons, 0, sps.shape[0]]
    plot_profile(sps[:, ons:one], extent=extent)

    return ons, one


def get_sps_fft(data, plot=False):
    """
    Function to perform LRFS (fourier transfrom along the singlu pulse axis)
    of the pulse stack to look for drifiting.

    Returns the fourier frequencies and the 2D power shape=(frequencies x len(data))

    Parameters:
    -----------------------------------------
    data: array
        The stack of single pulses, 2D array
    plot: bool
        Flag to decide to show the results
    -----------------------------------------
    """

    # Check for nans
    mask = np.isnan(np.median(data, 1)) | (np.std(data, 1) == 0)
    data[mask] = np.median(data[~mask], 0)

    # Make it zero mean
    data += -np.nanmean(data, 0)

    # Do FFT
    ft = np.fft.rfft(data, axis=0)
    freq = np.fft.rfftfreq(data.shape[0])[1:]
    power = (np.abs(ft) ** 2).T[:, 1:]

    if plot:
        fig, ax = plt.subplots(
            nrows=2,
            ncols=2,
            gridspec_kw={"width_ratios": [2, 1], "height_ratios": [2, 1]},
        )
        ax[1][1].remove()

        ax[0][0].imshow(power, origin="lower", aspect="auto", cmap="hot")
        # ax[0][0].set_ylabel("Phase bins")
        ax[0][0].set_xticks(ticks=[])
        ax[0][1].plot(
            power.sum(1) / np.max(power.sum(1)),
            np.arange(data.shape[1]),
            c="blue",
            alpha=0.8,
        )
        ax[0][1].set_xlabel("Normalized power")
        # ax[0][1].set_yticks(ticks=[])
        ax[1][0].plot(freq, power.sum(0) / np.max(power.sum(0)), c="red", alpha=0.8)
        ax[1][0].set_xlabel("Frequency (in units of (1/P))")
        ax[1][0].set_ylabel("Normalized power")
        plt.tight_layout()
        plt.show()
    return freq, power


def calc_fwhm(x, y):
    """
    Function to calculate the FWHM of the peak

    Given two array x and y, it calculates the range of x
    for which y is atleast half the extreme value. Returns the
    maximum value of y, FWHM and the left and right limits of x
    where y>=0.5(ymin+ymax)

    Parameters:
    -----------------------------------------
    x: array

    y: array
    -----------------------------------------
    """
    max_pow = np.argmax(y)
    fwhm_val = 0.5 * (np.max(y) + np.median(y))
    if max_pow == len(y) - 1:
        fwhm_r = np.nan
    else:
        fwhm_r = x[np.where(y[max_pow:] - fwhm_val <= 0)[0][0] + max_pow]
    if max_pow == 0:
        fwhm_l = 0
    else:
        fwhm_l = x[np.where(y[:max_pow] - fwhm_val <= 0)[0][-1]]
    fwhm = fwhm_r - fwhm_l
    return [max_pow, fwhm, fwhm_l, fwhm_r]


def get_nfprob_fft(nfprob, l=256, plot=False):
    """
    Function to looks for NF periodicity

    Takes the null probabilities of single pulses, divides into stacks of 256
    and looks for periodicity by adding power from all the stacks incoherently
    Returns the fourier frequencies, added power, analytical limit on the power
    assuming noise and FAR=1e-3 (used to assess the significance of the maximum)
    and the FWHM

    Parameters:
    -----------------------------------------
    nfprob: array
        The null probabilities of single pulses
    l: int
        The length of each stack
    plot: bool
        Flag to decide to show the results
    -----------------------------------------
    """
    # Get rid of the nans
    nfprob[np.isnan(nfprob)] = np.nanmedian(nfprob)
    # Make it zero mean, we are not interested in 0 frequency
    nfprob -= np.mean(nfprob)

    # Get high resolution FFT
    hresfft = np.fft.rfft(nfprob)
    hresfr = np.fft.rfftfreq(len(nfprob))
    hrespow = np.abs(hresfft) ** 2 / np.max(np.abs(hresfft) ** 2)

    # Get low res FFT
    nchuncks = len(nfprob) // l
    lresfr = np.fft.rfftfreq(l)

    if nchuncks > 0:
        power = []
        for i in range(nchuncks):
            chunck = nfprob[l * i : l * (i + 1)]
            chunck -= np.mean(chunck)

            # Take FFT
            lresfft = np.fft.rfft(chunck)
            am, bm = lresfft.real, lresfft.imag
            am *= 1 / np.std(am)
            bm *= 1 / np.std(bm)
            power.append(am**2 + bm**2)

        power = np.array(power)
        comb_pow_norm = np.sum(power, 0)

    # Calculate FWHM
    fwhm = calc_fwhm(lresfr[1:], comb_pow_norm[1:])

    # Calculate theoretical limit
    # Get analytical limit using chi2 distribution
    # Define FAR cut-off assuming trail factor
    pcut = (1 - 0.001) ** (2 / l)
    analyt_lim = chi2.ppf(df=2 * nchuncks, q=pcut)

    if plot:
        plt.clf()
        for ind, i in enumerate(power):
            if ind == 0:
                plt.plot(
                    lresfr[1:],
                    i[1:] * (np.max(comb_pow_norm) * 0.3 / np.max(i)),
                    color="gray",
                    alpha=0.25,
                    label="Individual stacks",
                )
            else:
                plt.plot(
                    lresfr[1:],
                    i[1:] * (np.max(comb_pow_norm) * 0.3 / np.max(i)),
                    color="gray",
                    alpha=0.25,
                )
        plt.plot(lresfr[1:], comb_pow_norm[1:], label="low res. FFT")
        plt.axvspan(xmin=fwhm[2], xmax=fwhm[3], color="g", alpha=0.5)
        plt.xlabel("Fourier frequency (in 1/P)", fontsize=20)
        plt.ylabel("Power (arbitrary units)", fontsize=20)
        plt.legend(fontsize=15)
        plt.tight_layout()
        plt.show()

    return lresfr, comb_pow_norm, analyt_lim, fwhm


def get_limits(nfprob, trials=1000, nch=20, l=256):

    """Function to get the upper and lower limits in the spectral power baed on FAR

    Does this emperically.Takes random draws from the null probability histogram and
    calculates maximum power. Returns frequency dependent upper and lower limits.

    Parameters:
    -----------------------------------------
    nfprob: array
        The null probabilities of single pulses
    trails: int
        The number of trails, significance will be 1 - 1/trails
    nch: int
        Number of stacks of data used to add incoherently to get the power
    l: int
        Length of each stack
    -----------------------------------------
    """

    nfprobdist, nfprobbins = get_nf_prob_dist(nf_prob=nfprob)
    nfprobdist *= 1 / nfprobdist.sum()

    lim = []

    # Now generate random draws
    for i in range(trials):
        comb_pow = []
        for j in range(nch):
            randdraw = np.random.choice(nfprobbins, size=l, p=nfprobdist)
            ft = np.fft.rfft(randdraw)
            am, bm = ft.real, ft.imag
            am *= 1 / np.std(am)
            bm *= 1 / np.std(bm)
            comb_pow.append(am**2 + bm**2)
            # comb_pow.append(np.abs(ft) ** 2 / (np.max(np.abs(ft) ** 2)))
        comb_pow = np.array(comb_pow)
        lim.append(comb_pow.sum(0))

    lim = np.array(lim)
    uplim = np.max(lim, 0)
    lolim = np.min(lim, 0)
    return uplim, lolim
