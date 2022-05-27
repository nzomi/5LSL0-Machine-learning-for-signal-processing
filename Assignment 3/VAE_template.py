# %% imports
from matplotlib.pyplot import axis
import torch
import torch.nn as nn

from main_template import load_model

ex8 = True

# %%  Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # create layers here
        self.conv_first = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1,bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(16,16,3,padding=1,bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        if ex8 is False:
            self.flatten1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*2*2,2),
            nn.ReLU()
            )
            self.flatten2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*2*2,2),
            nn.ReLU()
            )
        else:
            self.flatten1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*2*2,16),
            nn.ReLU()
            )
            self.flatten2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*2*2,16),
            nn.ReLU()
            )
        self.n = torch.distributions.Normal(0,1)
        self.n.loc = self.n.loc.cuda()
        self.n.scale = self.n.scale.cuda()
        self.kl = 0

    def forward(self, x):
        # use the created layers here
        # x --> 32,32,1
        x = self.conv_first(x) # 16,16,16
        x = self.conv(x) # 8,8,16
        x = self.conv(x) # 4,4,16
        x = self.conv(x) # 2,2,16
        miu = self.flatten1(x) # 64 --> 2
        sigma = torch.exp(self.flatten2(x)) # 64 --> 2
        h = miu + sigma * self.n.sample(miu.shape)
        self.kl = 0.5 * (sigma + miu**2 - torch.log(sigma) - 1).sum()
        return miu, sigma, h
    
# %%  Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # create layers here
        if ex8 is False:
            self.flatten = nn.Sequential(
            nn.Linear(2,16*2*2),
            nn.ReLU()
            )
        else:
            self.flatten = nn.Sequential(
            nn.Linear(16,16*2*2),
            nn.ReLU()
            )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(16,16,3,padding=1,bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2,mode='nearest')
        )
        self.conv_last = nn.Sequential(
            nn.ConvTranspose2d(16,1,3,padding=1,bias=True),
            nn.BatchNorm2d(1),
            nn.Upsample(scale_factor=2,mode='nearest')
        )
        
    def forward(self, x):
        # use the created layers here
        # x --> 1,2
        x = self.flatten(x) # x --> 1,64
        x = x.reshape(-1,16,2,2) # 2,2,16
        x = self.conv(x) # 4,4,16
        x = self.conv(x) # 8,8,16
        x = self.conv(x) # 16,16,16
        h = self.conv_last(x) # 32,32,1
        return h
    
# %%  Autoencoder
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, x):
        a, b, h = self.encoder(x)
        r = self.decoder(h)
        return h, r
        
#%%
def sanity_check():

    x = torch.randn((64,1,32,32)).cuda()

    model1, model2 = Encoder(), Decoder()
    model3 = VAE()
    for model in [model1,model2,model3]:
        model = model.to(torch.device('cuda:0'))
    miu, sigma, h = model1(x)
    out = model2(h)
    latent, r = model3(x)

    print('Encoder',miu.shape) 
    print('Encoder',sigma.shape) 
    print('Encoder',h.shape) 
    print('Decoder',out.shape)
    print('VAE',latent.shape)
    print('VAE',r.shape)

# %%
if __name__ == "__main__":
    sanity_check()
# %%
