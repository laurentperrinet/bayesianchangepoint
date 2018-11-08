# -*- coding: utf8 -*-
from __future__ import print_function, division
"""
BBCP: Binomial Bayesian Changepoint Detection

    url='https://github.com/laurentperrinet/bayesianchangepoint',

 An implementation of:
 @TECHREPORT{ adams-mackay-2007,
    AUTHOR = {Ryan Prescott Adams and David J.C. MacKay},
    TITLE  = "Bayesian Online Changepoint Detection",
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

    if Jeffreys: from scipy.stats import beta
    np.random.seed(seed)

    trials = np.arange(N_trials)
    p = np.random.rand(N_trials, N_blocks, N_layer)
    for trial in trials:
        # drawing all switches
        p[trial, :, 2] = np.random.rand(1, N_blocks) < 1./tau
        if Jeffreys:
            p_random = beta.rvs(a=.5, b=.5, size=N_blocks)
        else:
            p_random = np.random.rand(1, N_blocks)
        # drawing all probability biases
        p[trial, :, 1] = (1 - p[trial, :, 2])*p[trial-1, :, 1] + p[trial, :, 2] * p_random
        # drawing all samples
        p[trial, :, 0] = p[trial, :, 1] > np.random.rand(1, N_blocks)

    return (trials, p)


def likelihood(o, p, r):
    """
    Knowing $p$ and $r$, the sufficient statistics of the beta distribution $B(\alpha, \beta)$ :
    $$
        alpha = p*r
        beta  = (1-p)*r
    $$
    the likelihood of observing o=1 is that of a binomial of

        - mean rate of chosing hypothesis "o=1" = (p*r + o)/(r+1)
        - number of choices where  "o=1" equals to p*r+1

    since both likelihood sum to 1, the likelihood of drawing o in the set {0, 1}
    is equal to

    """
    L =  (1-o) * ( 1 - 1 / (p * r + 1) )**(p*r) * ((1-p) * r + 1) + o * ( 1 - 1 / ((1-p) * r + 1) )**((1-p)*r) * (p * r + 1)
    L /=         ( 1 - 1 / (p * r + 1) )**(p*r) * ((1-p) * r + 1) +     ( 1 - 1 / ((1-p) * r + 1) )**((1-p)*r) * (p * r + 1)

    return L

def prior(p):
    """

    https://en.wikipedia.org/wiki/Jeffreys_prior

    """
    return (p * (1-p)) ** -.5


def inference(o, h, p0=.5, r0=1., verbose=False, max_T=None):
    """
    Args:
      * o (np.ndarray): data has given in a sequence of observations as a
        function of (discrete) time (or trials). The total number of trials
        is T.

      * h (float): hazard rate, a value in the interval [0, 1] that is the
        probability of a changepoint at any given time.

      * p0, r0 (float, float): specify prior beta-distribution for p.
        This data is Binomial with unknown mean.  We are going to
        use the standard conjugate prior, that is, a beta-ditribution.
        The beta-ditribution yields a closed-form
        predictive distribution, which makes it easy to use in this context.
        The prior on r0 takes the value $1$ for a Jeffrey prior and $2$ for a
        uniform prior.

    Output:
      * beliefs (np.ndarray): beliefs about the run lengths at a given trial, the first
          axis (one row) is the probability vector at any given time. This vector
          is of length at maximum T (the maximal run length or ``max_T`` if
          specified). It represents the infered probability for each given run-length
          hypothesis for the given trial given the past observations.

            - the first axis records the estimated prebabilities
            for the different hypothesis of run lengths
            - the second axis is time (trials) - the system has only access to the present
            time, but this is a convenience for plots. Outputs give the inference
            for the current trial, before the actual observation.

      * p_bar (np.ndarray): mean of the prediction about p. Given the run-lengths r,
          this gives the sufficient statistics for our belief about p (second
          layer) for any coming trial. Has the same dimension as ``beliefs``.

    """
    if not max_T is None:
        # unless otherwise specified
        T = max_T
    else:
        # max is by default the total number of observations
        T = o.size

    # check parameter range
    assert(0 <= h <= 1)
    assert(0 <= p0 <= 1)

    # First, setup the matrix that will hold our beliefs about the current
    # run lengths.  We'll initialize it all to zero at first.
    beliefs = np.zeros((T, T))

    # INITIALIZATION
    # At time t=0, we actually have complete knowledge about the possible run
    # length probabilities. It is surely zero: the corresponding probability
    # is 1 at r=0 and zero elsewhere.
    beliefs[0, 0] = 1.0

    # Track the current set of parameters.  These start out at the prior and
    # we accumulate data as we proceed.
    p_bar = np.zeros((T, T))
    p_bar[0, 0] = p0

    # matrix of r=alpha+beta in the beta-ditribution
    r_bar = np.zeros((T, T))
    r_bar[0, 0] = r0

    # Loop over the data
    for t in range(T-1):
        # EVALUATION
        # we use the knowledge at time t to evaluate the likelihood of the new datum o[t]

        # Evaluate the predictive distribution for the next datum assuming that
        # we know the sufficient statistics of the pdf that generated the datum.
        # This probability is computed over the set of possible run-lengths.
        pi_hat = likelihood(o[t], p_bar[:(t+1), t], r_bar[:(t+1), t])

        if verbose and t<8:
            print('time', t, '; obs=', o[t], '; beliefs=', beliefs[:(t+1), t], '; pi_hat=', pi_hat, '; 1-h=', (1-h), '; p_bar=', p_bar[:(t+1), t])

        # PREDICTION
        # we use prior knowledge about the generative model to predict the state
        # of the system at time t+1

        # 1/ Evaluate the growth probabilities at time t+1
        # it is a vector for the belief of the different run-length
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

        # 2/ Update the sufficient statistics for each possible run length.
        # the vector of the different run-length at trial t+1
        # it has size t+2 to represent all possible run lengths
        r_bar[1:(t+2), t+1] = r_bar[:(t+1), t] + 1
        r_bar[0, t+1] = r0
        # the corresponding mean
        p_bar[1:(t+2), t+1] = p_bar[:(t+1), t] * r_bar[:(t+1), t] / r_bar[1:(t+2), t+1]
        p_bar[1:(t+2), t+1] += o[t] / r_bar[1:(t+2), t+1]
        p_bar[0, t+1] = p0


    return p_bar, r_bar, beliefs

def readout(p_bar, r_bar, beliefs, mode='expectation', fixed_window_size=40):
    """
    Retrieves a readout given a probabilistic representation

    Different modes are available:
    - 'expectation' : gives the average value of the estimated
    - 'max' : gives the most liklelikely values at each time,
    - 'hindsight' : looks back in time knowing a complete inference run
    - 'fixed' : considers a fixed Window

    """
    modes = ['expectation', 'max', 'fixed', 'hindsight']
    N_trials = beliefs.shape[-1]
    if mode in modes:
        if mode=='expectation':
            p_hat = np.sum(p_bar * r_bar * beliefs, axis=0)
            r_hat = np.sum(r_bar * beliefs, axis=0)
            p_hat[r_hat > 0] /= r_hat[r_hat > 0]
            p_hat[r_hat==0] = .5
            # r_hat = np.sum(r * beliefs, axis=0)[:-1]
        elif mode=='max':
            r_ = np.argmax(beliefs, axis=0)
            p_hat = np.array([p_bar[r_[i], i] for i in range(N_trials)])
            r_hat = np.array([r_bar[r_[i], i] for i in range(N_trials)])
        elif mode=='fixed':
            r_ = np.zeros(N_trials, dtype=np.int)
            for i in range(N_trials):
                if i <= fixed_window_size :
                    r_[i] = i
                else :
                    r_[i] = fixed_window_size
            p_hat = np.array([p_bar[r_[i], i] for i in range(N_trials)])
            r_hat = np.array([r_bar[r_[i], i] for i in range(N_trials)])
        elif mode=='hindsight':
            p_hat = np.zeros(N_trials)
            r_hat = np.zeros(N_trials)
            # initialize to the last measure
            idx = 0
            # propagate backwards
            for t in range(N_trials)[::-1]:
                if idx == 0 :
                    idx = np.argmax(beliefs[:, t])
                    p_hat[t] = p_bar[idx, t]
                    r_hat[t] = r_bar[idx, t]
                else :
                    idx -= 1
                    p_hat[t] = p_hat[t+1]
                    r_hat[t] = r_hat[t+1] - 1

        return p_hat, r_hat
    else:
        print ('mode ', mode, 'must be in ', modes)
        return None

def plot_inference(o, p_true, p_bar, r_bar, beliefs, mode='expectation', fixed_window_size=40, fig=None, axs=None, fig_width=13, max_run_length=120, eps=1.e-12, margin=0.01):
    import matplotlib.pyplot as plt

    N_trials = o.size
    if fig is None:
        fig_width= fig_width
        fig, axs = plt.subplots(2, 1, figsize=(fig_width, fig_width/1.6180), sharex=True)

    axs[0].step(range(N_trials), o, lw=1, alpha=.9, c='k')
    if not p_true is None:
        axs[0].step(range(N_trials), p_true, lw=3, alpha=.4, c='b')

    p_hat, r_hat = readout(p_bar, r_bar, beliefs, mode=mode, fixed_window_size=fixed_window_size)

    from scipy.stats import beta
    p_low, p_sup = np.zeros_like(p_hat), np.zeros_like(p_hat)
    for i_trial in range(N_trials):
        p_low[i_trial], p_sup[i_trial] = beta.ppf([.05, .95], a=p_hat[i_trial]*r_hat[i_trial], b=(1-p_hat[i_trial])*r_hat[i_trial])

    axs[0].plot(range(N_trials), p_hat, lw=1, alpha=.9, c='r')
    axs[0].plot(range(N_trials), p_sup, 'r--', lw=1, alpha=.9)
    axs[0].plot(range(N_trials), p_low, 'r--', lw=1, alpha=.9)
    if mode == 'fixed':
        axs[1].imshow(np.log(beliefs[:max_run_length, :]*0. + eps ))
    else:
        axs[1].imshow(np.log(beliefs[:max_run_length, :] + eps ))
    axs[1].plot(range(N_trials), r_hat, lw=1, alpha=.9, c='r')

    for i_layer, label in zip(range(2), ['p_hat +/- CI', 'belief on r = p(r)']):
        axs[i_layer].axis('tight')
        #axs[i_layer].set_xlim(0, N_trials+1)
        axs[i_layer].set_xticks(np.linspace(0, N_trials, 5, endpoint=True))
        axs[i_layer].set_xticklabels([str(int(k)) for k in np.linspace(0, N_trials, 5, endpoint=True)])
        axs[i_layer].set_ylim(-margin, 1 + margin)
        # axs[i_layer].axis('tight')
#            axs[i_layer].set_yticks(np.arange(1)+.5)
#            axs[i_layer].set_yticklabels(np.arange(1) )
        axs[i_layer].set_ylabel(label, fontsize=14)
        #
    axs[-1].set_xlabel('trials', fontsize=14);
    axs[-1].set_ylim(0, max_run_length);
    fig.tight_layout()

    return fig, axs
