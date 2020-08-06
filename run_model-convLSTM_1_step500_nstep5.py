import time
import tqdm
import torch
import numpy as np

import chnet.cahn_hill as ch

from chnet.lstm_unet import LSTM_Unet
from chnet.ch_loader import CahnHillDataset
from toolz.curried import pipe, curry, compose
from torch.utils.data import Dataset, DataLoader


@curry
def mse_loss(y1, y2, scale=1.):
    """standard MSE definition"""
    assert y1.shape == y2.shape
    return ((y1 - y2) ** 2).sum() / y1.data.nelement() * scale


@curry
def init_norm(nsamples, dim_x, dim_y, seed=354875, m_l=-0.15, m_r=0.15):
    np.random.seed(seed)
    means  = np.random.uniform(m_l, m_r, size=nsamples)
    np.random.seed(seed)
    scales  = np.random.uniform(0.05, 0.3, size=nsamples)
    
    x_data = [np.random.normal(loc=m, scale=s, size = (1, dim_x, dim_y)) for m,s in zip(means, scales)]
    x_data = np.concatenate(x_data, axis=0)
    
    np.clip(x_data, -0.98, 0.98, out=x_data)
    
    return x_data


@curry
def data_generator(nsamples=2, 
                   dim_x=64, 
                   init_steps=1, 
                   delta_sim_steps = 100,
                   dx = 0.25, 
                   dt = 0.01,
                   gamma=1.0, 
                   m_l=-0.15, 
                   m_r=0.15,
                   seed = 354875,
                   device = torch.device("cuda:0")):
    
    init_data = init_norm(nsamples, dim_x, dim_x, seed=seed, m_l=m_l, m_r=m_r)   

    x_data = ch.ch_run_torch(init_data, 
                             dt=dt, gamma=gamma, 
                             dx=dx, sim_step=init_steps, 
                             device=device)    

    y_data = ch.ch_run_torch(x_data, 
                             dt=dt, gamma=gamma, 
                             dx=dx, sim_step=delta_sim_steps, 
                             device=device)    
    return x_data, y_data


@curry
def data_generator_slices(nsamples=2, 
                   dim_x=64, 
                   init_steps=1, 
                   delta_sim_steps = 100,
                   dx = 0.25, 
                   dt = 0.01,
                   gamma=1.0, 
                   seed = None,
                   n_step = 4,
                   m_l=-0.15, 
                   m_r=0.15,
                   device = torch.device("cuda:0")):
    
    init_data = init_norm(nsamples, dim_x, dim_x, seed=seed, m_l=m_l, m_r=m_r)   

    x_list = []
    y_list = []
    
    x_data = ch.ch_run_torch(init_data, 
                             dt=dt, gamma=gamma, 
                             dx=dx, sim_step=init_steps, 
                             device=device)    
    

    for _ in range(n_step):
        x_list.append(x_data[None])
        x_data = ch.ch_run_torch(x_data, 
                                 dt=dt, gamma=gamma, 
                                 dx=dx, sim_step=delta_sim_steps, 
                                 device=device)
        y_list.append(x_data[None])
    
    x_data = np.moveaxis(np.concatenate(x_list, axis=0), 0, 1)
    y_data = np.moveaxis(np.concatenate(y_list, axis=0), 0, 1)
    
    return x_data, y_data


# transformer_x = lambda x: x[None]
# transformer_y = lambda x: x[None]
transformer_x = lambda x: x[:,None]
transformer_y = lambda x: x[:,None]

def main():
    device = torch.device("cuda:0")
    
    dim_x=96
    
    init_steps=1
    n_step = 5
    delta_sim_steps=500
    
    dx = 0.25 
    dt = 0.01
    gamma = 1.0
    
    m_l = -0.15, 
    m_r = 0.15

    n_samples_trn = 1024*7

    num_epochs = 10     

    ngf = 32    
    log_step = 512
        
    batch_size=8
    fname = "out/unet_convLSTM_ch{}_init_{}_delta_{}_nstep_{}.pt".format(ngf, 
                                                                  init_steps, 
                                                                  delta_sim_steps, 
                                                                  n_step)
    print("File tag: ", fname)
    
    model = LSTM_Unet(input_nc=1, 
                      output_nc=1, 
                      ngf=ngf, 
                      temporal=n_step, 
                      tanh=True).double().to(device)
    
    criterion = mse_loss(scale=10000)
    learning_rate = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=learning_rate)
    
    trn_losses = []
    ##############################################################    
    
    count = 0
    seeds = [23513*29, 124081*17, 1982981*31, 1783*19, 
             8574*23, 2364*93, 871523*113, 8961*43, 
             98712*73, 14112*31, 785384*93, 12434*113, 
             429785*117]
    
    for seed_trn in seeds:
        
        count += 1
        print("Generating Training Data, turn: {}".format(count))  
        x_trn, y_trn = data_generator_slices(nsamples=n_samples_trn, 
                                            dim_x=dim_x, 
                                            init_steps=init_steps, 
                                            delta_sim_steps = delta_sim_steps,
                                            dx = dx, 
                                            dt = dt,
                                            m_l=-0.15, 
                                            m_r=0.15,
                                            n_step=n_step,
                                            gamma=gamma, 
                                            seed =seed_trn,
                                            device = device)


        trn_dataset = CahnHillDataset(x_trn, y_trn, 
                                      transform_x=transformer_x, 
                                      transform_y=transformer_y)
        
        trn_loader = DataLoader(trn_dataset, 
                                batch_size=batch_size, 
                                shuffle=True, 
                                num_workers=4)

        print("Starting Model Training")
        for epoch in range(num_epochs):    

            for i, item_trn in enumerate(tqdm.tqdm(trn_loader)):       
                model.train()

                x = item_trn['x'].to(device)
                y_t = item_trn['y'].to(device)

                # Forward pass
                y_p, y_p1 = model(x)
                loss = criterion(y_t, y_p) + criterion(y_t, y_p1)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                trn_losses.append(np.sqrt(loss.item()))

                if (i) % log_step == 0:                 
                    print ('Epoch [{}/{}], Step [{}/{}], Training Loss: {:.11f}'.format(epoch+1, 
                                                                                        num_epochs, 
                                                                                        i+1, 
                                                                                        len(trn_loader), 
                                                                                        np.mean(trn_losses[-log_step:])))

            torch.save(model.state_dict(), fname)

if __name__ == "__main__":
    main()
