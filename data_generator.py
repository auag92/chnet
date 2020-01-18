import os
import time
import torch
import argparse
import numpy as np
import chnet.cahn_hill as ch
from toolz.curried import pipe, curry, compose, memoize
    

def init_unif(nsamples, dim_x, dim_y,scale=0.1, seed=354875):
    np.random.seed(seed)
    return (2 * np.random.random((nsamples, dim_x, dim_y)) - 1) * scale


def init_norm(nsamples, dim_x, dim_y, scale_mean=0.1, scale_std=0.1, seed=354875):
    np.random.seed(seed)
    means  = (2 * np.random.random(nsamples) - 1) * scale_mean
    x_data = [np.random.normal(loc=m, scale=scale_std, size = (1, dim_x, dim_y)) for m in means]
    x_data = np.concatenate(x_data, axis=0)
    return x_data

def get_fname(output_folder, indx, args):
    return output_folder + "/ch_%d_gamma_%d_dt_%d_dx_%d_%d_%s.npy" % (indx*sim_steps, 
                                                                      int(args.gamma*1000), 
                                                                      int(args.dt*1000), 
                                                                      int(args.dx*1000), 
                                                                      args.dim_x, args.init)

def generate(nsamples=4096, dim_x=101, dim_y=101, 
             sim_steps=100, n_runs=100, dx=0.25, 
             dt=0.01, gamma=1.0, device=torch.device("cuda:0"), 
             init="unif", infile=None, output_folder="indata"):

    if init == "unif":
        x_data = init_unif(nsamples, dim_x, dim_y, seed=354875)
    elif init == "norm":
        x_data = init_norm(nsamples, dim_x, dim_y, seed=784361)
    
    print("Starting Execution")
    
    y_data = torch.from_numpy(x_data).double().to(device)
    fname = output_folder + "/ch_%d_gamma_%d_dt_%d_dx_%d_%d_%s.npy" % (0, 
                                                                       int(gamma*1000), 
                                                                       int(dt*1000), 
                                                                       int(dx*1000), 
                                                                       dim_x, init)
    
    
    np.save(get_fname(output_folder, 0, dt, dx, init), y_data.cpu().numpy())
    
    for i in range(n_runs):

        
        start = time.time()
        y_data = ch.ch_run_torch(y_data, dt=dt, gamma=gamma, dx=dx, sim_step=sim_steps, device=device)
        elpsd = time.time() - start
        
        print("{}, memory:{}, {}".format(elpsd, torch.cuda.memory_allocated(device), fname))        


        np.save(fname, y_data.cpu().numpy())

    print("Done Simulation %s" % (fname))
    print("Ending Execution")
        
if __name__ == "__main__":
    main()
