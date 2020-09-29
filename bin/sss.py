#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
   This file belong to https://github.com/snolfi/evorobotpy
   and has been written by Stefano Nolfi, stefano.nolfi@istc.cnr.it
   requires es.py, policy.py, and evoalgo.py 
"""

import numpy as np
from numpy import zeros, dot, sqrt
import math
import time
from evoalgo import EvoAlgo
from utils import descendent_sort


# Evolve with SSS
class SSS(EvoAlgo):
    def __init__(self, env, policy, seed, filedir):
        EvoAlgo.__init__(self, env, policy, seed, filedir)

    def save(self, ceval, cgen, maxsteps, bfit, bgfit, avefit, aveweights):
            print('save data')
            # save best postevaluated so far
            fname = self.filedir + "/bestgS" + str(self.seed)
            if (self.policy.normalize == 0):
                np.save(fname, self.bestgsol)
            else:
                np.save(fname, np.append(self.bestgsol,self.policy.normvector))
            # save best so far
            fname = self.filedir + "/bestS" + str(self.seed)
            if (self.policy.normalize == 0):
                np.save(fname, self.bestsol)
            else:
                np.save(fname, np.append(self.bestsol,self.policy.normvector))  
            # save statistics
            fname = self.filedir + "/statS" + str(self.seed)
            np.save(fname, self.stat)
            # save summary statistics
            fname = self.filedir + "/S" + str(self.seed) + ".fit"
            fp = open(fname, "w")
            fp.write('Seed %d (%.1f%%) gen %d msteps %d bestfit %.2f bestgfit %.2f cbestfit %.2f cbestgfit %.2f avgfit %.2f weightsize %.2f \n' %
                      (self.seed, ceval / float(maxsteps) * 100, cgen, ceval / 1000000, self.bestfit, self.bestgfit, bfit, bgfit, avefit, aveweights))
            fp.close()

    def run(self, maxsteps):

        start_time = time.time()               # start time
        nparams = self.policy.nparams          # number of parameters
        popsize = self.batchSize               # popsize
        ceval = 0                              # current evaluation
        cgen = 0                               # current generation
        rg = np.random.RandomState(self.seed)  # create a random generator and initialize the seed
        pop = rg.randn(popsize, nparams)       # population
        fitness = zeros(popsize)               # fitness
        fitness_beh = zeros((popsize, 3))
        self.stat = np.arange(0, dtype=np.float64) # initialize vector containing performance across generations

        assert ((popsize % 2) == 0), print("the size of the population should be odd")

        # initialze the population
        for i in range(popsize):
            pop[i] = self.policy.get_trainable_flat()       

        print("SSS: seed %d maxmsteps %d popSize %d noiseStdDev %lf crossoverrate %lf nparams %d" % (self.seed, maxsteps / 1000000, popsize, self.noiseStdDev, self.crossoverrate, nparams))

        # main loop
        elapsed = 0
        while (ceval < maxsteps):
            
            cgen += 1

            # If normalize=1 we update the normalization vectors
            if (self.policy.normalize == 1):
                self.policy.nn.updateNormalizationVectors()
                
            self.env.seed(self.policy.get_seed + cgen)        # set the environment seed, it changes every generation
            self.policy.nn.seed(self.policy.get_seed + cgen)  # set the policy seed, it changes every generation
            
            # Evaluate the population
            for i in range(popsize):                           
                self.policy.set_trainable_flat(pop[i])        # set policy parameters
                eval_rews, eval_length, rews1, rews2 = self.policy.rollout(self.policy.ntrials, timestep_limit=1000)  # evaluate the individual
                fitness[i] = eval_rews                        # store fitness
                fitness_beh[i] = np.array([i, rews1, rews2])
                ceval += eval_length                          # Update the number of evaluations
                self.updateBest(fitness[i], pop[i])           # Update data if the current offspring is better than current best

            fitness, index = descendent_sort(fitness)         # create an index with the ID of the individuals sorted for fitness
            bfit = fitness[index[0]]
            self.updateBest(bfit, pop[index[0]])              # eventually update the genotype/fitness of the best individual so far

            # PARETO-FRONT
            pareto_front_idx = []
            front_len = []
            halfpopsize = int(popsize/2)
            #dominated = fitness_beh.copy()
            count = 0
            #while len(dominated) > 0:
            #current_level = []
            current_idx = []
            for i in range(len(fitness_beh)):
                res = ~(fitness_beh[i] > fitness_beh)
                res = np.delete(res, i, axis=0)
                if not (np.any(np.all(res, axis=1))):
                    #current_level.append(dominated[i])
                    pareto_front_idx.append(int(fitness_beh[i, 0]))
                    current_idx.append(i)
                    count += 1
                    # if len(pareto_front_idx) == halfpopsize:
                    #    break
            print("Number of genotypes in the pareto-front: %.2f" %(count))
            #pareto_front_idx.append(current_level)
            #front_len.append(len(current_idx))
            dominated = np.array(np.delete(fitness_beh[:, 0], current_idx, axis=0), dtype=np.int64)
            
            childrensize = popsize-count
            parent = np.random.choice(pareto_front_idx, size=childrensize, replace=True)
            cross_prob = np.random.uniform(low=0.0, high=1.0, size=childrensize)

            for i in range(childrensize):
                # crossover of the first parent and a randomly selected second parent among the first pareto-front
                if cross_prob[i] < self.crossoverrate:
                    parent_1 = pop[parent[i]]
                    idx_p2 = np.random.choice(pareto_front_idx, size=2, replace=False)
                    if idx_p2[0] != parent[i]:
                        parent_2 = pop[idx_p2[0]]
                    else:
                        parent_2 = pop[idx_p2[1]]
                    cutting_points = np.random.choice(np.arange(0, nparams, 1), size=2, replace=False)
                    min_point = cutting_points.min()
                    max_point = cutting_points.max()
                    
                    # The section A and C of the first parent with the section B of the second parent  
                    if np.random.uniform(low=0.0, high=1.0) < 0.5:
                        pop[dominated[i], :min_point] = parent_1[:min_point]
                        pop[dominated[i], min_point:max_point] = parent_2[min_point:max_point]
                        pop[dominated[i], max_point:] = parent_1[max_point:]
                    # The section A and C of the second parent with the section B of the first parent
                    else:
                        pop[dominated[i], :min_point] = parent_2[:min_point]
                        pop[dominated[i], min_point:max_point] = parent_1[min_point:max_point]
                        pop[dominated[i], max_point:] = parent_2[max_point:]
                        
                    pop[dominated[i]] += (rg.randn(nparams) * self.noiseStdDev)

                else:
                    pop[dominated[i]] = pop[parent[i]] + (rg.randn(1, nparams) * self.noiseStdDev)


            # Postevaluate the best individual
            self.env.seed(self.policy.get_seed + 100000)      # set the environmental seed, always the same for the same seed
            self.policy.nn.seed(self.policy.get_seed + 100000)# set the policy seed, always the same for the same seed
            self.policy.set_trainable_flat(pop[index[0]])     # set the parameters of the policy
            eval_rews, eval_length, _, _ = self.policy.rollout(self.policy.ntrials, timestep_limit=1000, post_eval=True)
            bgfit = eval_rews
            ceval += eval_length
            self.updateBestg(bgfit, pop[index[0]])            # eventually update the genotype/fitness of the best post-evaluated individual

            # display info
            print('Seed %d (%.1f%%) gen %d msteps %d bestfit %.2f bestgfit %.2f cbestfit %.2f cbestgfit %.2f avgfit %.2f weightsize %.2f' %
                      (self.seed, ceval / float(maxsteps) * 100, cgen, ceval / 1000000, self.bestfit, self.bestgfit, bfit, bgfit, np.average(fitness), np.average(np.absolute(pop[index[0]]))))

            # store data throughout generations
            self.stat = np.append(self.stat, [ceval, self.bestfit, self.bestgfit, bfit, bgfit, np.average(fitness)])

            # save data
            if ((time.time() - self.last_save_time) > (self.policy.saveeach * 60)):
                self.save(ceval, cgen, maxsteps, bfit, bgfit, np.average(fitness), np.average(np.absolute(pop[index[0]])))
                self.last_save_time = time.time()  

        self.save(ceval, cgen, maxsteps, bfit, bgfit, np.average(fitness), np.average(np.absolute(pop[index[0]])))
        end_time = time.time()
        print('Simulation time: %dm%ds ' % (divmod(end_time - start_time, 60)))