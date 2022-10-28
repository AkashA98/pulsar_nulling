# Pulsar Nulling

A mixture model for analyzing nulling in pulsars, improved on [Kaplan+2018](https://github.com/dlakaplan/nulling-pulsars). Given the ON and OFF histograms, it uses [emcee](http://dfm.io/emcee/current/) to do Markov Chain Monte Carlo fit to the ON/OFF histograms to estimate the nulling fraction. Currently, it can fit for any number of components if they are Gaussian and only 2 components (1 for OFF and 1 for ON) if the ON distribution is describes by [exponentially modified Gaussian](https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution).

## It can be installed via pip:

```git clone git@github.com:AkashA98/pulsar_nulling.git
cd pulsar_nulling/
pip install -e .
```

## Example of the code usage:

If the user has the ON and OFF window intensities as arrays

```
from nulling.get_nf import nulls
nf = nulls(on, off, ncomp=2, model='exp_tail', nwalkers=32, burnin=200, nsteps=5000)
nf.run_mcmc(corr=True)
```
If the user wants to take into account the finite correlation length of the chain in order to get independent samples, then they can set the ```corr``` flag to True. All the results are available as class attributes. For example, the best fit values can be obtained by ```nf.fit_val``` and the errors can be obtained by ```nf.fit_err```. The ordering or parameters will always be 
### [[means], [standard deviations], [tau(optional)], [weights]]
