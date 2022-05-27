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
no_epochs = 30
learning_rate = 3e-4

# get dataloader
train_loader, test_loader = MNIST_dataloader.create_dataloaders(data_loc, batch_size)

# create the autoencoder
model = autoencoder_template.AE()
# create the optimizer
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
criterion = nn.MSELoss()

# %% training loop
# go over all epochs
def train(model,optimizer,criterion,epochs,file_name):
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
            loss = criterion(score,x_clean)
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
                loss = criterion(score,x_clean)
                loss_val += loss.item()

        
        train_loss.append(loss_train/len(train_loader))
        val_loss.append(loss_val/len(test_loader))
        loss_train = 0.0
        loss_val = 0.0

    torch.save(model, file_name + str(epochs) + '.pth')

    return model, latent, score, train_loss, val_loss

#%%
def test_model(model,data_loader):
    train_list = []
    latent_list = []
    label_list = []
    model.eval()
    with torch.no_grad():
            for batch_idx,(x_clean, x_noisy, label) in enumerate(tqdm(data_loader)):
                # fill in how to train your network using only the clean images
                if torch.cuda.is_available():
                    device = torch.device('cuda:0')
                    x_clean, x_noisy, label = [x.cuda() for x in [x_clean, x_noisy, label]]
                    model.to(device)
                latent, score = model(x_clean)
                x_clean, latent, label = [x.detach().cpu() for x in [x_clean,latent,label]]
                train_list.append(x_clean)
                latent_list.append(latent)
                label_list.append(label)

    return train_list, latent_list, label_list

#%%
# move back to cpu    
# get some examples
def image_plot(model):
    examples = enumerate(test_loader)
    _, (x_clean_example, x_noisy_example, labels_example) = next(examples)
    latent, score= model(x_clean_example.cuda())
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
    # plt.savefig("Fig/Exercise_1_image.png",dpi=300,bbox_inches='tight')
    plt.show()  

#%%
def loss_plot(train_loss, val_loss):
    plt.figure(figsize=(12,6))
    plt.plot(train_loss,label = "Train loss")
    plt.plot(val_loss, label = "Test loss")
    plt.title('Train and Test Losses')
    plt.xlabel('Epoch[n]')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("Fig/Exercise_1_loss.png",dpi=300,bbox_inches='tight')

#%%
def load_model(filename):
    """ Load the trained model.
    Args:
        model (Model class): Untrained model to load.
        filename (str): Name of the file to load the model from.
    Returns:
        Model: Model with parameters loaded from file.
    """
    model = torch.load(filename)
    return model

#%%
def dataloader_type_change(model,dataloader):
    x_clean, x_noisy, labels = test_model(model,dataloader)
    x_clean = torch.cat(x_clean, dim = 0)
    x_noisy= torch.cat(x_noisy, dim = 0)
    labels = torch.cat(labels, dim = 0)
    return x_clean, x_noisy, labels

def get_latent(x_clean, labels, model):
    latent, score = model(x_clean)
    latent = latent.data.cpu().numpy()
    labels = torch.Tensor.numpy(labels)
    return latent, labels

#%%
def scatter_plot(model):
    x_clean_test  = test_loader.dataset.Clean_Images
    x_noisy_test  = test_loader.dataset.Noisy_Images
    labels_test   = test_loader.dataset.Labels
    latent, score= model(x_clean_test.cuda())
    score = score.data.cpu().numpy()
    latent = latent.data.cpu().numpy().reshape(-1,1,2,1)
    fig,ax = plt.subplots(figsize=(16,9))
    scatter = ax.scatter(latent[:,0,0,0],latent[:,0,1,0],c=labels_test)
    legend = ax.legend(*scatter.legend_elements(),loc='upper right',title='digits')
    ax.add_artist(legend)
    plt.title('latent scatter')
    plt.show()

# %% HINT
#hint: if you do not care about going over the data in mini-batches but rather want the entire dataset use:
# x_clean_train = train_loader.dataset.Clean_Images
# x_noisy_train = train_loader.dataset.Noisy_Images
# labels_train  = train_loader.dataset.Labels

x_clean_test  = test_loader.dataset.Clean_Images
x_noisy_test  = test_loader.dataset.Noisy_Images
labels_test   = test_loader.dataset.Labels

# use these 10 examples as representations for all digits
# x_clean_example = x_clean_test[0:10,:,:,:]
# x_noisy_example = x_noisy_test[0:10,:,:,:]
# labels_example = labels_test[0:10]

#%%
if __name__ == "__main__":
    # model, latent, score, train_loss, val_loss = train(model, optimizer, criterion, 25, 'AE_')
    # image_plot(model)
    # loss_plot(train_loss, val_loss)
    model = load_model('AE_25.pth')
    latent_test, labels_test = get_latent(x_clean_test.cuda(),labels_test,model)
    # latent_train, labels_train = get_latent(x_clean_train.cuda(),labels_test,model)
    scatter_plot(latent_test,labels_test)


# %%
