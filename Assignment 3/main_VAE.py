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
from main_template import load_model, loss_plot

# %% set torches random seed
torch.random.manual_seed(0)

# %% preperations
# define parameters
data_loc = '5LSL0-Datasets' #change the data location to something that works for you
batch_size = 64
no_epochs = 30
learning_rate = 1e-3

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
    loss_test = 0.0
    train_loss = []
    test_loss = []
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
                loss_test += loss.item()

        if epoch%5 ==0:
            print(f'train_loss = {loss_train/len(train_loader)}, test_loss = {loss_test/len(test_loader)}')

        train_loss.append(loss_train/len(train_loader))
        test_loss.append(loss_test/len(test_loader))
        loss_train = 0.0
        loss_test = 0.0

    torch.save(model, file_name + str(epochs) + '.pth')

    return model, latent, score, train_loss, test_loss

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
    #plt.savefig("Fig/Exercise_7_image.png",dpi=300,bbox_inches='tight')
    plt.show()  

#%%
def mnistgrid_plot(model):
    x_grid = np.linspace(-2,2,15)
    y_grid = np.linspace(-2,2,15)[::-1] # from top to buttom
    x_sample, y_sample = np.meshgrid(x_grid,y_grid)
    latent_sample = np.stack((x_sample.flatten(),y_sample.flatten()),axis=1)
    latent_sample_tensor = torch.from_numpy(latent_sample).float().reshape(-1,2)
    decoder = model.decoder(latent_sample_tensor.cuda())
    decoder_grid = decoder.data.cpu().numpy()
    x_idx, y_idx = 15, 15

    # plot the images in a grid
    plt.figure(figsize=(12, 12))
    for j in range(y_idx):
        for i in range(x_idx):
            img_idx = i + j * x_idx + 1
            plt.subplot(x_idx, y_idx, img_idx)
            plt.imshow(decoder_grid[img_idx-1, 0, :, :], cmap='gray')
            plt.xticks([])
            plt.yticks([])

    plt.tight_layout()
    #plt.savefig("Fig/Exercise_7_grid.png", dpi=300, bbox_inches='tight')
    plt.show()


# %%
if __name__ == '__main__':
    model, latent, score, train_loss, test_loss = train(model, optimizer, 10, 'VAE_')
    # model = load_model('VAE_50.pth')
    loss_plot(train_loss,test_loss)
    image_plot(model)
    mnistgrid_plot(model)
# %%
