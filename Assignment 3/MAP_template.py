# %% imports
# libraries
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from main_template import load_model, image_plot
import MNIST_dataloader

# %% MAP Estimation
# parameters
no_iterations = 1000
learning_rate = 1e-2
beta = 0.01
data_loc = '5LSL0-Datasets' #change the data location to something that works for you
batch_size = 64

# get dataloader
train_loader, test_loader = MNIST_dataloader.create_dataloaders(data_loc, batch_size)

model = load_model('VAE_ex8_50.pth')
estimated_latent = nn.Parameter(torch.randn(10,16))
optimizer_map = torch.optim.Adam([estimated_latent],lr = learning_rate)

examples = enumerate(test_loader)
_, (x_clean_example, x_noisy_example, labels_example) = next(examples)


#%%
running_loss = []
# optimization
for i in tqdm(range(no_iterations)):
    optimizer_map.zero_grad()
    score = model.decoder(estimated_latent.cuda())
    loss = ((x_noisy_example[0:10,:,:,:].cuda() - score)**2).sum() + (beta*estimated_latent).sum()
    loss.backward()
    optimizer_map.step()
    running_loss.append(loss.item()/1e3)
    if i%100==0:
        print(f'loss = {loss.item()}')

#%%

ex8_out = model.decoder(estimated_latent.cuda())
ex8_out = ex8_out.data.cpu().numpy()
print(ex8_out.shape)
# %%
plt.figure(figsize=(36,9))
for i in range(10):        
    plt.subplot(3,10,i+1)
    plt.imshow(x_noisy_example[i,0,:,:],cmap='gray')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3,10,i+11)
    plt.imshow(ex8_out[i,0,:,:],cmap='gray')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3,10,i+21)
    plt.imshow(x_clean_example[i,0,:,:],cmap='gray')
    plt.xticks([])
    plt.yticks([])

    plt.savefig("Fig/Exercise_8b_image.png", dpi=300, bbox_inches='tight')
plt.show()
# %%
plt.figure(figsize=(12,6))
plt.plot(running_loss,label = "Train loss")
plt.title('MAP Losses')
plt.xlabel('Iterations[n]')
plt.ylabel('Loss')
plt.legend()
plt.savefig("Fig/Exercise_8b_loss.png",dpi=300,bbox_inches='tight')
