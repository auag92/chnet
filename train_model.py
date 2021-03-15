import sys
import json
import numpy as np
from tqdm import tqdm
from toolz.curried import pipe, curry, compose

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

import chnet.ch_tools as ch_tools
import chnet.utilities as ch_utils
import chnet.ch_generator as ch_gen
from chnet.ch_loader import CahnHillDataset
from chnet.models import UNet, UNet_solo_loop, UNet_loop, mse_loss


def train(key="unet", mid=0.0, dif=0.449, dim_x=96, dx=0.25, dt=0.01, 
            gamma=0.2, init_steps=1, nstep=20, n_samples_trn=1024, 
            ngf=32, final_tstep = 5000, num_epochs=10, 
            learning_rate=1.0e-5, n_primes=2000, 
            device="cuda"):
    
    m_l=mid-dif
    m_r=mid+dif
    delta_sim_steps=(final_tstep-init_steps)//nstep
    primes = ch_utils.get_primes(n_primes)
    
    print("no. of datasets: {}".format(len(primes)))
    
    device = torch.device("cuda:0") if device == "cuda" else torch.device("cpu")
    print(device)
    if key == "unet":
        model=UNet(in_channels=1, out_channels=1, init_features=ngf).double().to(device)
    elif key == "unet_solo_loop":
        model=UNet_solo_loop(in_channels=1, out_channels=1, init_features=ngf, temporal=nstep).double().to(device)
    elif key == "unet_loop":
        model=UNet_loop(in_channels=1, out_channels=1, init_features=ngf, temporal=nstep).double().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    trn_losses = []

    fout = "weights/model_{}_size_{}_step_{}_init_{}_delta_{}_tstep_{}.pt".format(key, ngf, nstep, init_steps, delta_sim_steps, num_epochs*len(primes))  
    print("model saved at: {}".format(fout))

    print("Start Training")
    for num, prime in enumerate(primes):
        # Loss and optimizer
        torch.cuda.empty_cache()
        x_trn, y_trn = ch_gen.data_generator(nsamples=n_samples_trn, 
                                      dim_x=dim_x, 
                                      init_steps=init_steps, 
                                      delta_sim_steps = delta_sim_steps,
                                      dx=dx, 
                                      dt=dt,
                                      m_l=m_l, 
                                      m_r=m_r,
                                      n_step=nstep,
                                      gamma=gamma, 
                                      seed=2513*prime,
                                      device=device)


        trn_dataset = CahnHillDataset(x_trn, y_trn, 
                                      transform_x=lambda x: x[:,None], 
                                      transform_y=lambda x: x[:,None])

        trn_loader = DataLoader(trn_dataset, 
                                batch_size=8, 
                                shuffle=True, 
                                num_workers=4)

        print("Training Run: {}, prime: {}".format(num, prime))

        total_step = len(trn_loader)
        
        for epoch in range(num_epochs):  
            for i, item_trn in enumerate(tqdm(trn_loader)):
                
                model.train()
                
                if "loop" in key:
                    if "solo" in key:
                        x = item_trn['x'][:,0].to(device)
                    else:
                        x = item_trn['x'][:,0].to(device)
                    y_tru = item_trn['y'].to(device)
                else:
                    x = item_trn['x'][:,0].to(device)
                    y_tru = item_trn['y'][:,-1] .to(device) 
                    
                y_prd = model(x) # Forward pass
                loss = mse_loss(y_tru, y_prd, scale=10000)
               
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                trn_losses.append(np.sqrt(loss.item()))

            print ('Epoch [{}/{}], Training Loss: {:.11f}'.format(epoch+1, num_epochs, np.mean(trn_losses[-total_step:])))

        obj = {}
        obj["state"] = model.state_dict()
        obj["losses"] = trn_losses
        torch.save(obj, fout)
    print("End Training")
    return model

if __name__ == "__main__":
    # arguments = {"key":"unet_solo_loop",
    #              "mid":0.0, 
    #              "dif":0.449, 
    #              "dim_x":96,
    #              "init_steps":1, 
    #              "dx":0.25,
    #              "dt":0.01,
    #              "gamma":0.2, 
    #              "n_samples_trn":1024,
    #              "ngf":8,
    #              "nstep":5,
    #              "final_tstep":5000,
    #              "num_epochs":10,
    #              "n_primes":5,
    #              "learning_rate":1.0e-4,
    #              "device":"cuda"}
    # for finp in ["args_loop.json", "args_solo_loop.json", "args_unet.json",]:
    #     with open(finp, 'r') as f:
    #         arguments = json.load(f)
    #     arguments["n_primes"] = 100
        # print(arguments)
    if len(sys.argv) > 1:
        for argv in sys.argv[1:]:
            with open(argv, 'r') as f:
                arguments = json.load(f)
            print(argv)
            print(arguments)
            model = train(**arguments)
    else:
        print("please supply input arguments file.")
