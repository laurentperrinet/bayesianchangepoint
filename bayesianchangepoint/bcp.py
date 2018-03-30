# -*- coding: utf8 -*-
from __future__ import print_function, division
"""

    url='https://github.com/laurentperrinet/bayesianchangepoint',

 An implementation of:
 @TECHREPORT{ adams-mackay-2007,
    AUTHOR = {Ryan Prescott Adams and David J.C. MacKay},
    TITLE  = "{B}ayesian Online Changepoint Detection",
    INSTITUTION = "University of Cambridge",
    ADDRESS = "Cambridge, UK",
    YEAR = "2007",
    NOTE = "arXiv:0710.3742v1 [stat.ML]"
 }

 for a binomial input.

 adapted from
    url='https://github.com/JackKelly/bayesianchangepoint',

 by
    Copyright 2013 Jack Kelly (aka Daniel) jack@jack-kelly.com
    author='Jack Kelly',
    author_email='jack@jack-kelly.com',

which is itself adapted from the matlab code @

    http://hips.seas.harvard.edu/content/bayesian-online-changepoint-detection


"""
import numpy as np

def switching_binomial_motion(N_trials, N_blocks, tau, seed, Jeffreys=True, N_layer=3):
    """

    A 3-layered model for generating samples.

    about Jeffrey's prior : see https://en.wikipedia.org/wiki/Jeffreys_prior

    """

    from scipy.stats import beta
    np.random.seed(seed)

    trials = np.arange(N_trials)
    p = np.random.rand(N_trials, N_blocks, N_layer)
    for trial in trials:
        p[trial, :, 2] = np.random.rand(1, N_blocks) < 1/tau # switch
        if Jeffreys: #
            p_random = beta.rvs(a=.5, b=.5, size=N_blocks)
        else:
            p_random = np.random.rand(1, N_blocks)
        p[trial, :, 1] = (1 - p[trial, :, 2])*p[trial-1, :, 1] + p[trial, :, 2] * p_random # probability
        p[trial, :, 0] =  p[trial, :, 1] > np.random.rand(1, N_blocks) # binomial

    return (trials, p)

def likelihood(o, p, r):
    """
    Knowing p and r, the likelihood of observing o is that of a binomial of

        - mean rate of chosing 1 = (p*r + o)/(r+1)
        - number of choices = 1 equal to p*r+1

    since both likelihood sum to 1, the likelihood of drawing o in {0, 1}
    is equal to

    """
    L =  (1-o) * (1 - 1 / (p * r + 1) )**(p*r) * ((1-p) * r + 1) + o * (1 - 1 / ((1-p) * r + 1) )**((1-p)*r) * (p * r + 1)
    L /=         (1 - 1 / (p * r + 1) )**(p*r) * ((1-p) * r + 1) +     (1 - 1 / ((1-p) * r + 1) )**((1-p)*r) * (p * r + 1)
    return L

def inference(o, h, p0=.5, r0=.5, verbose=False, max_T=None):
    """
    Args:
      * o (np.ndarray): data has given in a sequence of observations as a
        function of (dicscrete) time (or trials). The totla number of trials
        is T.

      * h (float): hazard rate, a value in the interval [0,1] that is the
        probability of a changepoint at any given time.

      * p0, r0 (float, float): specify initial values for beta-distribution.

      * alpha0, beta0 (float, float): specify prior beta-distribution for p.
        This data is Binomial with unknown mean.  We are going to
        use the standard conjugate prior of a beta-ditribution. ** Note that
        one cannot use non-informative priors for changepoint detection in
        this construction.  The beta-ditribution yields a closed-form
        predictive distribution, which makes it easy to use in this context. **

    Output:
      * beliefs (np.ndarray): beliefs about the current run lengths, the first
          axis (one row) is the probability vector at any given time. This vector
          is of length at maximum T( the maximal run length). It represents the
          probability of a given run-length after one observation.

      * p_bar (np.ndarray): mean of the prediction about p. Given the run-lengths r,
          this gives the sufficient statistics for our belief about p at any given
          time.

            - the first axis records the estimated prebabilities
            for the different hypothesis of run lengths
            - the second axis is time (trials) - the system has only access to the present
            time, but this is a convenience for plots.

    """
    if max_T is None:
        T = o.size # total number of observations
    else:
        T = max_T
    # First, setup the matrix that will hold our beliefs about the current
    # run lengths.  We'll initialize it all to zero at first.
    beliefs = np.zeros((T+1, T+1))

    # INITIALIZATION
    # At time t=0, we actually have complete knowledge about the possible run
    # length probabilities. It is zero: the corresponding probability is 1 at 0
    # and zero elsewhere.
    beliefs[0, 0] = 1.0

    # Track the current set of parameters.  These start out at the prior and
    # we accumulate data as we proceed.
    p_bar = np.zeros((T+1, T+1))
    p_bar[0, 0] = p0

    # matrix of run lengths
    r = np.zeros((T+1, T+1))

    # Loop over the data like we're seeing it all for the first time.
    for t in range(T):
        # the vector of the different run-length at time t+1
        # it has size t+2 to represent all possible run lengths
        r[:(t+1), t] = np.arange(0, t+1) + r0

        # Evaluate the predictive distribution for the next datum assuming that
        # we know the sufficient statistics of the pdf that generated the datum.
        # This probability is computed over the set of possible run-lengths.
        pi_hat = likelihood(o[t], p_bar[:(t+1), t], r[:(t+1), t])

        if verbose and t<8:
            print('time', t, '; obs=', o[t], '; beliefs=', beliefs[:(t+1), t], '; pi_hat=', pi_hat, '; 1-h=', (1-h), '; p_bar=', p_bar[:(t+1), t])
        # Evaluate the growth probabilities at
        # it is a vector for the belief of the different run-length at time t+1
        # knowing the datum observed until time t
        # it has size t+2 to represent all possible run lengths up to time t along with the new datum
        belief = np.zeros((t+2))
        # iff there was no changepoint, shift the probabilities up in the graph
        # scaled by the hazard function and the predictive
        # probabilities.
        belief[1:] = beliefs[:(t+1), t] * pi_hat * (1-h)

        # Evaluate the probability that there *was* a changepoint and we're
        # accumulating the mass back down at a run length of 0.
        belief[0] = np.sum(beliefs[:(t+1), t] * pi_hat * h)
        #if verbose and t <8: print('belief=', belief)
        # Renormalize the run length probabilities by calculating total evidence
        belief = belief / np.sum(belief)
        # record this vector
        beliefs[:(t+2), t+1] = belief
        if verbose and t <8: print('Note that at t', t, ', belief', belief[0], '= h = ', h)

        # Update the sufficient statistics for each possible run length.
        p_bar[1:(t+2), t+1] = p_bar[:(t+1), t] * r[:(t+1), t] / (r[:(t+1), t] + 1)
        p_bar[1:(t+2), t+1] += o[t] / (r[:(t+1), t] + 1)
        p_bar[0, t+1] = p0
        # for i in range(1, t+2):
        #     #if verbose and t <8: print(t, i, r[i, t]+1, o[(t-i+1):(t+1)])
        #     p_bar[i, t+1] = np.mean( o[(t-i+1):(t+1)] ) #/ (r[i, t] +1)
        #
    return p_bar, r, beliefs



def readout(p_bar, r, beliefs, mode='expectation', fixed_window_size=40):
    """
    Retrieves a readout given a probabilistic representation

    Different modes are available:
    - 'expectation' : gives the average value of the estimated
    - 'max' : gives the most liklelikely values at each time,

    """
    if mode=='expectation':
        p_hat = np.sum(p_bar[:, 1:] * beliefs[:, :-1], axis=0)
        r_hat = np.sum(r * beliefs, axis=0)[:-1]
    elif mode=='max':
        belief_max = np.argmax(beliefs, axis=0)[:-1]
        p_hat = np.array([p_bar[belief_max[i], i+1] for i in range(belief_max.size)])
        r_hat = belief_max
    elif mode=='fixed':
        r_hat=[]
        for i in range(len(p_bar)-1):
            if i <= fixed_window_size :
                r_hat.append(i)
            else :
                r_hat.append(fixed_window_size)
        p_hat = np.array([p_bar[r_hat[i], i+1] for i in range(len(r_hat))])
    # TODO : implement elif mode=='hindsight':
    return p_hat, r_hat

def plot_inference(o, p_true, p_bar, r, beliefs, mode='expectation', fixed_window_size=40, fig=None, axs=None, fig_width=13, max_run_length=120):
    import matplotlib.pyplot as plt
    N_trials = o.size

    if fig is None:
        fig_width= fig_width
        fig, axs = plt.subplots(2, 1, figsize=(fig_width, fig_width/1.6180), sharex=True)

    axs[0].step(range(N_trials), o, lw=1, alpha=.9, c='k')
    if not p_true is None:
        axs[0].step(range(N_trials), p_true, lw=1, alpha=.9, c='b')

    p_hat, r_hat = readout(p_bar, r, beliefs, mode=mode, fixed_window_size=fixed_window_size)
    from scipy.stats import beta
    p_low, p_sup = np.zeros_like(p_hat), np.zeros_like(p_hat)
    for i_trial in range(N_trials):
        p_low[i_trial], p_sup[i_trial] = beta.ppf([.05, .95], a=p_hat[i_trial]*r_hat[i_trial], b=(1-p_hat[i_trial])*r_hat[i_trial])

    axs[0].plot(range(N_trials), p_hat, lw=1, alpha=.9, c='r')
    axs[0].plot(range(N_trials), p_sup, 'r--', lw=1, alpha=.9)
    axs[0].plot(range(N_trials), p_low, 'r--', lw=1, alpha=.9)
    axs[1].imshow(np.log(beliefs[:max_run_length, :] + 1.e-5 ))
    axs[1].plot(range(N_trials), r_hat, lw=1, alpha=.9, c='r')

    for i_layer, label in zip(range(2), ['p_hat +/- CI', 'belief on r = p(r)']):
        axs[i_layer].set_xlim(0, N_trials)
        axs[i_layer].set_ylim(-.05, 1 + .05)
        axs[i_layer].axis('tight')
#            axs[i_layer].set_yticks(np.arange(1)+.5)
#            axs[i_layer].set_yticklabels(np.arange(1) )
        axs[i_layer].set_ylabel(label, fontsize=14)
        axs[i_layer].axis('tight')
    axs[-1].set_xlabel('trials', fontsize=14);
    fig.tight_layout()

    return fig, axs
