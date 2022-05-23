# %% imports
# libraries
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

# local imports
import MNIST_dataloader
import autoencoder_template

# %% set torches random seed
torch.random.manual_seed(0)

# %% preperations
# define parameters
data_loc = '5LSL0-Datasets' #change the data location to something that works for you
batch_size = 64
no_epochs = 50
learning_rate = 1e-3

# get dataloader
train_loader, test_loader = MNIST_dataloader.create_dataloaders(data_loc, batch_size)

# create the autoencoder
model = autoencoder_template.AE()

# create the optimizer
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
criterion = nn.MSELoss()
train_loss = []
# %% training loop
# go over all epochs
for epoch in range(no_epochs):
    print(f"\nTraining Epoch {epoch}:")
    # go over all minibatches
    for batch_idx,(x_clean, x_noisy, label) in enumerate(tqdm(train_loader)):
        # fill in how to train your network using only the clean images
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            x_clean, x_noisy, label = [x.cuda() for x in [x_clean, x_noisy, label]]
            model.to(device)
        latent, score = model(x_clean)
        loss = criterion(score,x_clean)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    print('Loss',np.mean(np.array(train_loss)))

#%%
# move back to cpu    
# get some examples
examples = enumerate(test_loader)
_, (x_clean_example, x_noisy_example, labels_example) = next(examples)
latent, score = model(x_clean_example.cuda())
latent = latent.data.cpu().numpy()
score = score.data.cpu().numpy()
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
#plt.savefig("exercise_1.png",dpi=300,bbox_inches='tight')
plt.show() 

#%%

plt.plot(train_loss)

# %% HINT
# #hint: if you do not care about going over the data in mini-batches but rather want the entire dataset use:
# x_clean_train = train_loader.dataset.Clean_Images
# x_noisy_train = train_loader.dataset.Noisy_Images
# labels_train  = train_loader.dataset.Labels

# x_clean_test  = test_loader.dataset.Clean_Images
# x_noisy_test  = test_loader.dataset.Noisy_Images
# labels_test   = test_loader.dataset.Labels

# # use these 10 examples as representations for all digits
# x_clean_example = x_clean_test[0:10,:,:,:]
# x_noisy_example = x_noisy_test[0:10,:,:,:]
# labels_example = labels_test[0:10]