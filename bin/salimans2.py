#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
   This file belong to https://github.com/snolfi/evorobotpy
   and has been written by Stefano Nolfi and Paolo Pagliuca, stefano.nolfi@istc.cnr.it, paolo.pagliuca@istc.cnr.it

   salimans.py include an implementation of the OpenAI-ES algorithm described in
   Salimans T., Ho J., Chen X., Sidor S & Sutskever I. (2017). Evolution strategies as a scalable alternative to reinforcement learning. arXiv:1703.03864v2

   requires es.py, policy.py, and evoalgo.py 

"""

import numpy as np
from numpy import zeros, dot, sqrt
import math
import time
from evoalgo import EvoAlgo
from utils import ascendent_sort
from scipy import stats

# Evolve with ES algorithm taken from Salimans et al. (2017)
class Salimans2(EvoAlgo):
    def __init__(self, env, policy, seed, filedir):
        EvoAlgo.__init__(self, env, policy, seed, filedir)

    def run(self, maxsteps):

        start_time = time.time()

        # initialize the solution center
        center = self.policy.get_trainable_flat()
        
        # Extract the number of parameters
        nparams = self.policy.nparams
        # setting parameters
        batchSize = self.batchSize
        if batchSize == 0:
            # 4 + floor(3 * log(N))
            batchSize = int(4 + math.floor(3 * math.log(nparams)))
        # Symmetric weights in the range [-0.5,0.5]
        weights = zeros(batchSize)

        ceval = 0                    # current evaluation
        cgen = 0                # current generation
        # Parameters for Adam policy
        m = zeros(nparams)
        v = zeros(nparams)
        epsilon = 1e-08 # To avoid numerical issues with division by zero...
        beta1 = 0.9
        beta2 = 0.999
    
        # RandomState for perturbing the performed actions (used only for samples, not for centroid)
        rs = np.random.RandomState(self.seed)
        fitbestsample = [0,0]

        print("Salimans2: seed %d maxmsteps %d batchSize %d stepsize %lf noiseStdDev %lf wdecay %d sameEnvCond %d nparams %d" % (self.seed, maxsteps / 1000000, batchSize, self.stepsize, self.noiseStdDev, self.wdecay, self.sameenvcond, nparams))


        # main loop
        elapsed = 0
        while (ceval < maxsteps):
            cgen += 1


            # Extract half samples from Gaussian distribution with mean 0.0 and standard deviation 1.0
            samples = rs.randn(batchSize, nparams)
            # buffer vector for candidate
            candidate = np.arange(nparams, dtype=np.float64)
            # allocate the fitness vector (fitness2 is the sum on the two behaviors)
            fitness = zeros(batchSize * 2)
            fitness2 = zeros(batchSize * 2)
            # If normalize=1 we update the normalization vectors
            if (self.policy.normalize == 1):
                self.policy.nn.updateNormalizationVectors()
            # Reset environmental seed every generation
            self.env.seed(self.policy.get_seed + cgen)
            self.policy.nn.seed(self.policy.get_seed + cgen)
            # Evaluate offspring 2 times (on behavior 1 and 2)
            g1 = 0.0
            g2 = 0.0
            for beh in range(2):
                for b in range(batchSize):
                    for bb in range(2):
                        if (bb == 0):
                            candidate = center + samples[b,:] * self.noiseStdDev
                        else:
                            candidate = center - samples[b,:] * self.noiseStdDev                            
                        # Set policy parameters 
                        self.policy.set_trainable_flat(candidate) 
                        # Evaluate the offspring
                        eval_rews, eval_length, rews1, rews2 = self.policy.rollout(self.policy.ntrials, seed=(self.seed + (cgen * self.batchSize) + b), timestep_limit=beh)
                        # store the fitness
                        fitness[b*2+bb] = eval_rews
                        fitness2[b*2+bb] += (eval_rews / 2.0)
                        # Update the number of evaluations
                        ceval += eval_length

                # Sort by fitness and compute weighted mean into center
                fitness, index = ascendent_sort(fitness)
                fitbestsample[beh] = fitness[batchSize * 2 - 1]
                # Now me must compute the symmetric weights in the range [-0.5,0.5]
                utilities = zeros(batchSize * 2)
                for i in range(batchSize * 2):
                    utilities[index[i]] = i
                utilities /= (batchSize * 2 - 1)
                utilities -= 0.5
                # Now we assign the weights to the samples
                for i in range(batchSize):
                    idx = 2 * i
                    weights[i] = (utilities[idx] - utilities[idx + 1]) # pos - neg
                i = 0
                if (beh == 0):
                    while i < batchSize:
                        gsize = -1
                        if batchSize - i < 500:
                            gsize = batchSize - i
                        else:
                            gsize = 500
                        g1 += dot(weights[i:i + gsize], samples[i:i + gsize,:]) # weights * samples
                        i += gsize
                    g1 /= (batchSize * 2)
                else:
                     while i < batchSize:
                        gsize = -1
                        if batchSize - i < 500:
                            gsize = batchSize - i
                        else:
                            gsize = 500
                        g2 += dot(weights[i:i + gsize], samples[i:i + gsize,:]) # weights * samples
                        i += gsize                       
                     g2 /= (batchSize * 2)                    

            # sum the gradient computed on behavior 1 and 2
            glob = g1 + g2

            # Weight decay
            if (self.wdecay == 1):
                globalg = -glob + 0.005 * center
            else:
                globalg = -glob

            # Sort by using the sum of the fitness obtained on the two behaviors
            fitness2, index = ascendent_sort(fitness2)
            centroidfit = 0
            if (self.policy.nttrials > 0):
                bestsamid = index[batchSize * 2 - 1]
                if ((bestsamid % 2) == 0):
                    bestid = int(bestsamid / 2)
                    candidate = center + samples[bestid] * self.noiseStdDev
                else:
                    bestid = int(bestsamid / 2)
                    candidate = center - samples[bestid] * self.noiseStdDev

                # Update data if the current offspring is better than current best
                self.updateBest(fitness2[bestsamid], candidate)
                # post-evaluate the best sample to compute the generalization
                self.env.seed(self.policy.get_seed + 100000)
                self.policy.nn.seed(self.policy.get_seed + 100000)
                self.policy.set_trainable_flat(candidate) 
                eval_rews, eval_length, rews1, rews2 = self.policy.rollout(self.policy.nttrials, timestep_limit=2, post_eval=True)
                gfit = eval_rews
                ceval += eval_length
                # eveltually store the new best generalization individual
                self.updateBestg(gfit, candidate)


            # ADAM policy
            # Compute how much the center moves
            a = self.stepsize * sqrt(1.0 - beta2 ** cgen) / (1.0 - beta1 ** cgen)
            m = beta1 * m + (1.0 - beta1) * globalg
            v = beta2 * v + (1.0 - beta2) * (globalg * globalg)
            dCenter = -a * m / (sqrt(v) + epsilon)
            # update center
            center += dCenter

            # Compute the elapsed time (i.e., how much time the generation lasted)
            elapsed = (time.time() - start_time)

            # Update information
            self.updateInfo(cgen, ceval, fitness, center, centroidfit, fitness[batchSize * 2 - 1], elapsed, maxsteps)
            corr = stats.pearsonr(g1,g2)
            print('Seed %d (%.1f%%) gen %d msteps %d bestfit %.2f bestgfit %.2f bestsam %.2f (%.1f %.1f) avg %.2f weightsize %.2f gradientcorr %.2f' %
                      (self.seed, ceval / float(maxsteps), cgen, ceval / 1000000, self.bestfit, self.bestgfit, fitness2[batchSize * 2 - 1], fitbestsample[0], fitbestsample[1], np.average(fitness2), np.average(np.absolute(center)), corr[0]))

            # Save centroid and associated vectors
            if (self.saveeachg > 0 and cgen > 0):
                if ((cgen % self.saveeachg) == 0):
                    # save best, bestg, and stat
                    self.save(cgen, ceval, centroidfit, center, fitness[batchSize * 2 - 1], (time.time() - start_time)) 
                    # save summary statistics
                    fname = self.filedir + "/S" + str(self.seed) + ".fit"
                    fp = open(fname, "w")
                    fp.write('Seed %d gen %d msteps %d bestfit %.2f bestgfit %.2f bestsam %.2f (%.2f %.2f) avg %.2f weightsize %.2f gradientcorr %.2f \n' %
                        (self.seed, cgen, ceval / 1000000, self.bestfit, self.bestgfit, fitness2[batchSize * 2 - 1], fitbestsample[0], fitbestsample[1], np.average(fitness2), np.average(np.absolute(center)), corr[0]))
                    fp.close()

        # save best, bestg, and stat
        self.save(cgen, ceval, centroidfit, center, fitness[batchSize * 2 - 1], (time.time() - start_time))
        # save summary statistics
        fname = self.filedir + "/S" + str(self.seed) + ".fit"
        fp = open(fname, "w")
        fp.write('Seed %d gen %d msteps %d bestfit %.2f bestgfit %.2f bestsam %.2f (%.2f %.2f) avg %.2f weightsize %.2f gradientcorr %.2f \n' %
                (self.seed, cgen, ceval / 1000000, self.bestfit, self.bestgfit, fitness2[batchSize * 2 - 1], fitbestsample[0], fitbestsample[1], np.average(fitness2), np.average(np.absolute(center)), corr[0]))
        fp.close()

        # print simulation time
        end_time = time.time()
        print('Simulation time: %dm%ds ' % (divmod(end_time - start_time, 60)))



