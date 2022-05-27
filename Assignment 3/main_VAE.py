#%%
from matplotlib import axis
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

# local imports
import MNIST_dataloader
import VAE_template
from main_template import load_model

# %% set torches random seed
torch.random.manual_seed(0)

# %% preperations
# define parameters
data_loc = '5LSL0-Datasets' #change the data location to something that works for you
batch_size = 64
no_epochs = 30
learning_rate = 3e-4

# get dataloader
train_loader, test_loader = MNIST_dataloader.create_dataloaders(data_loc, batch_size)

# create the autoencoder
model = VAE_template.VAE()
# create the optimizer
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

# %% training loop
# go over all epochs
def train(model,optimizer,epochs,file_name):
    model.train()
    loss_train = 0.0
    loss_val = 0.0
    train_loss = []
    val_loss = []
    for epoch in range(epochs):
        # go over all minibatches
        for batch_idx,(x_clean, x_noisy, label) in enumerate(tqdm(train_loader)):
            # fill in how to train your network using only the clean images
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
                x_clean, x_noisy, label = [x.cuda() for x in [x_clean, x_noisy, label]]
                model.to(device)
            latent, score = model(x_clean)
            loss = ((score-x_clean)**2).sum() + model.encoder.kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        model.eval()
        with torch.no_grad():
            for batch_idx,(x_clean, x_noisy, label) in enumerate(tqdm(test_loader)):
                # fill in how to train your network using only the clean images
                if torch.cuda.is_available():
                    device = torch.device('cuda:0')
                    x_clean, x_noisy, label = [x.cuda() for x in [x_clean, x_noisy, label]]
                    model.to(device)
                latent, score = model(x_clean)
                loss = ((score-x_clean)**2).sum() + model.encoder.kl
                loss_val += loss.item()

        if epoch%5 ==0:
            print(f'train_loss = {loss_train/len(train_loader)}, test_loss = {loss_val/len(test_loader)}')

        train_loss.append(loss_train/len(train_loader))
        val_loss.append(loss_val/len(test_loader))
        loss_train = 0.0
        loss_val = 0.0

    torch.save(model, file_name + str(epochs) + '.pth')

    return model, latent, score, train_loss, val_loss

#%%
def image_plot(model):
    examples = enumerate(test_loader)
    _, (x_clean_example, x_noisy_example, labels_example) = next(examples)
    latent, score= model(x_clean_example.cuda())
    score = score.data.cpu().numpy()
    latent = latent.data.cpu().numpy().reshape(-1,1,2,1)
    # show the examples in a plot
    plt.figure(figsize=(12,3))
    for i in range(10):
        plt.subplot(3,10,i+1)
        plt.imshow(x_clean_example[i,0,:,:],cmap='gray')
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(3,10,i+11)
        plt.imshow(latent[i,0,:,:],cmap='gray')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(3,10,i+21)
        plt.imshow(score[i,0,:,:],cmap='gray')
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.savefig("Fig/Exercise_7_image.png",dpi=300,bbox_inches='tight')
    plt.show()  

# %%
if __name__ == '__main__':
    # model, latent, score, train_loss, val_loss = train(model, optimizer, 30, 'VAE_')
    model = load_model('VAE_30.pth')
    image_plot(model)
# %%
