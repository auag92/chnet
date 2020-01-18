import time
import tqdm
import torch
import numpy as np
import chnet.cahn_hill as ch
from chnet.ch_net import CHnet
from chnet.ch_loader import CahnHillDataset
from toolz.curried import pipe, curry, compose
from torch.utils.data import Dataset, DataLoader

def mse_loss(y1, y2, scale=1.):
    """standard MSE definition"""
    assert y1.shape == y2.shape
    return ((y1 - y2) ** 2).sum() / y1.data.nelement() * scale

@curry
def rmse_loss(y1, y2, scale=1.):
    """standard RMSE definition"""
    assert y1.shape == y2.shape
    return ((((y1 - y2) ** 2).sum() / y1.data.nelement()).sqrt()) * scale


def init_unif(nsamples, dim_x, dim_y, seed=354875):
    np.random.seed(seed)
    return np.random.uniform(-0.95, 0.95, size=(nsamples, dim_x, dim_y))


def init_norm(nsamples, dim_x, dim_y, seed=354875):
    np.random.seed(seed)
    means  = np.random.uniform(-0.1, 0.1, size=nsamples)
    scales  = np.random.uniform(0.1, 0.5, size=nsamples)
    
    x_data = [np.random.normal(loc=m, scale=s, size = (1, dim_x, dim_y)) for m,s in zip(means, scales)]
    x_data = np.concatenate(x_data, axis=0)
    
    np.clip(x_data, -0.95, 0.95, out=x_data)
    
    return x_data


def generate_data(size=12000, dim_x=101, sim_steps=6000, delta_t=100, seed=354875, device = torch.device("cuda:0")):
    dx = 0.25 # delta space_dim
    dt = 0.01 # delta time
    gamma = 1.0 # interface energy
    
    init_data1 = init_unif(size//2, dim_x, dim_x, seed)
    init_data2 = init_norm(size//2, dim_x, dim_x, (seed//1000) * 1982)
    init_data = np.concatenate([init_data1, init_data2], axis=0)
    
    x_data = ch.ch_run_torch(init_data, dt=dt, gamma=gamma, dx=dx, sim_step=sim_steps, device=device)
    y_data = ch.ch_run_torch(x_data, dt=dt, gamma=gamma, dx=dx, sim_step=delta_t, device=device)
    
    return x_data, y_data

@curry
def add_neighbors(x):
    dimx = x.shape[0]
    y = np.pad(x, pad_width=[[2,2],[2,2]], mode="wrap")
    out = [x[None]]
    for ix in [0, 1, 2, 3, 4]:
        for iy in [0, 1, 2, 3, 4]:
            out.append((y[ix:ix+dimx, iy:iy+dimx] * x)[None])
    return np.concatenate(out, axis=0)

def main():
    device = torch.device("cuda:0")
    print("Generating Training Data")
    x_train, y_train = generate_data(size=12000, sim_steps=6000, delta_t=100, seed=982354, device=device)

    print("Size of training set: ", x_train.shape, y_train.shape)
    print("Generating validation data")
    x_val, y_val = generate_data(size=100, sim_steps=6000, delta_t=100, seed=211354, device=device)
    print("Size of validation set: ", x_val.shape, y_val.shape)
    
    ks = 5 # kernel size
    in_channels = 26 # no. of input channels
    cw = 64 # channel width
    lx = (ks // 2) * 5 # pad width

    transformer_x = compose(lambda x: add_neighbors(x), 
                            lambda x: np.pad(x, pad_width=[[lx,lx],[lx,lx]], mode='wrap'))
    transformer_y = lambda x: x[None]

    train_dataset = CahnHillDataset(x_train, y_train, transform_x=transformer_x, transform_y=transformer_y)
    trainloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1)
    val_dataset = CahnHillDataset(x_val, y_val, transform_x=transformer_x, transform_y=transformer_y)

    total_step = len(trainloader)
    print("No. of training steps: %d" % total_step)
    
    model = CHnet(ks=ks, in_channels=in_channels, cw=cw).double().to(device)
    
    print("Printing Model Parameters:")
    nprod = 0
    for parameter in model.parameters():
        print(parameter.size())
        nprod += np.prod(parameter.size())
    print("No. of Parameters: %d" % nprod)
    
    num_epochs = 51
    criterion = rmse_loss(scale=10.)
    learning_rate = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    val_losses = []
    
    # Loss and optimizer
    for epoch in range(num_epochs):    
        if epoch % 2 == 0:
            torch.save(model.state_dict(), "weights/CH_trial_3_%d" % (epoch))
        if epoch > 1:
            learning_rate = 1e-6
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for i, item in enumerate(tqdm.tqdm(trainloader)):
            model.train()

            x = item['x'].to(device)
            target = item['y'].to(device)

            # Forward pass
            output = model(x)
            loss = criterion(output, target)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            if (i) % 100 == 0:
                for indx in np.random.permutation(np.arange(0, len(val_dataset)))[:5]:
                    model.eval()
                    item1 = val_dataset[indx]
                    x1 = item1['x'][None].to(device)
                    y1 = item1['y'][None].to(device)
                    # Forward pass
                    y2 = model(x1)
                    val_losses.append(criterion(y2, y1).item())

                print ('Epoch [{}/{}], Step [{}/{}], Training Loss: {:.11f}, Validation Loss: {:.11f}'.format(epoch+1, 
                                                                                                              num_epochs, 
                                                                                                              i+1, 
                                                                                                              total_step, 
                                                                                                          np.mean(train_losses[-50:]), 
                                                                                                              np.mean(val_losses[-5:])))
    np.save("training_losses.npy", train_losses)
    np.save("validation_losses.npy", val_losses)
    
if __name__ == "__main__":
    main()
