# %% imports
import torch
import torch.nn as nn

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
        self.conv_last = nn.Sequential(
            nn.Conv2d(16,1,2,padding=(2,1),bias=True),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
    def forward(self, x):
        # use the created layers here
        # x --> 32,32,1
        x = self.conv_first(x) # 16,16,16
        x = self.conv(x) # 8,8,16
        x = self.conv(x) # 4,4,16
        x = self.conv(x) # 2,2,16
        h = self.conv_last(x) # 1,2,1
        return h
    
# %%  Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # create layers here
        self.conv_first = nn.Sequential(
            nn.ConvTranspose2d(1,16,3,padding=1,bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Upsample(scale_factor=(1,2),mode='nearest')
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
        # x --> 1,2,1
        x = self.conv_first(x) # 2,2,16
        x = self.conv(x) # 4,4,16
        x = self.conv(x) # 8,8,16
        x = self.conv(x) # 16,16,16
        h = self.conv_last(x) # 32,32,1
        return h
    
# %%  Autoencoder
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, x):
        h = self.encoder(x)
        r = self.decoder(h)
        return h,r
        
#%%
def sanity_check():

    x = torch.randn((64,1,32,32))

    model_1, model_2, model = Encoder(), Decoder(), AE()

    latent = model_1(x)
    out = model_2(latent)
    y1, y2 = model(x)
    print('Encoder',latent.shape) # should be 64,16,2,1
    print('Decoder',out.shape) # should be 64,1,32,32
    print('AE_latent',y1.shape) # should be 64,16,2,1
    print('AE_output',y2.shape) # should be 64,1,32,32

# %%
if __name__ == "__main__":
    sanity_check()

# %%
