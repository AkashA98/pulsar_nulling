# Pulsar Nulling

A mixture model for analyzing nulling in pulsars, improved on [Kaplan+2018](https://github.com/dlakaplan/nulling-pulsars). In addition to the standard Gaussian mixture implemented by [Kaplan+2018](https://github.com/dlakaplan/nulling-pulsars), this extends the analysis to the cases where the distribution is more accurately described by an [exponentially modified Gaussian](https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution). This also allows to study the correlations in nulling, particularly looking for periodicities in the nulling patterns. Can also be used to study basic drifitng phenonmenon.

## It can be installed in the standard way as 
```
python ./setup.py install
```

## It can be installed via pip:

```
git clone git@github.com:AkashA98/pulsar_nulling.git
cd pulsar_nulling/
pip install .
```

## Example of the code usage:

If the user has the ON and OFF window intensities as arrays

```
from nulling.get_nf import nulls
nf = nulls(on, off, ncomp=2, model='exp_tail', nwalkers=32, burnin=200, nsteps=5000)
nf.run_mcmc(corr=True)
```
If the user wants to take into account the finite correlation length of the chain in order to get independent samples, then they can set the ```corr``` flag to True. All the results are available as class attributes. For example, the best fit values can be obtained by ```nf.fit_val``` and the errors can be obtained by ```nf.fit_err```. The ordering or parameters will always be 
##### [means, standard deviations, tau(optional), weights]

For example, in the case of a 2 component Gaussian model, they will be [mu1, mu2, std1, std2, nf] (Remember all the weights are not independent) and in the case of a 2 component exponentially modified gaussian, they will be [mu1, mu2, std1, std2, tau, nf]


The posteriors for the model parameters and the fit for the histograms can be plotted using

```
nf.plot_fit()
```

Comaprison of different models can be done using the Akaike information criterion [AIC](https://en.wikipedia.org/wiki/Akaike_information_criterion) and Bayesian information criterion [BIC](https://en.wikipedia.org/wiki/Bayesian_information_criterion) and can be obained by calling

```
nf.AIC()
nf.BIC()
```

and accessed as ```nf.aic_val``` and ```nf.bic_val```.

Once the nulling results are available, one can look for correlations in nulling. This depends on the null probabilities, the probability that a single pulse is null.

```
from nulling.correlations import *
NP = get_nf_prob(on, model="exp_tail", ndim=2, fit_val=nf.fit_val)
freq, power, lim, fwhm = get_nfprob_fft(NP)
```

where ```freq``` is the fourier frequencies, ```power``` is the stacked power from stacks of 256 single pulses (to mitigate ISM effects) and the ```lim``` is the analytical limit assuming FAR=0.001 under the null hypothesis of pure noise.

One can also look at the null and emission length histograms
```
null_len_dist, em_len_dist, null_len_tau, em_len_tau = get_null_em_dist(NP)
```
which gives the null and emission length distributions and the decay constant assuming an exponential fit.

Finally, one can look for drifitng patterns using the 2D pulse stack. If ```sps``` is the 2D array of the single pulses

```
freq, 2d_power = get_sps_fft(sps, plot=True)
```
