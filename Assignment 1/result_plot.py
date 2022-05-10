import numpy as np
from matplotlib import pyplot as plt

data_dir = './log/'    #data path containing the saved performance data
#choose the needed variables from below list
variable_name = ['train_loss_list',       #the loss value for each iteration of the training process
	             'val_loss_list',   #the loss value for each iteration of the val process
                ]
def plot_img(clean, noise, prediction, model_id):
    # show the examples in a plot
    plt.figure(figsize=(12,3))
    for i in range(10):
        plt.subplot(3,10,i+1)
        plt.imshow(clean[i,0,:,:],cmap='gray')
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(3,10,i+11)
        plt.imshow(noise[i,0,:,:],cmap='gray')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(3,10,i+21)
        plt.imshow(prediction[i,0,:,:],cmap='gray')
        plt.xticks([])
        plt.yticks([])
    
    plt.tight_layout()
    plt.savefig(model_id + 'prediction.png',dpi=300,bbox_inches='tight')

def plot_save(variable_name, data_dir):
    plt.figure(figsize=(12,12))
    for variable in variable_name:
        data = np.loadtxt(data_dir+variable+'.txt')
        plt.plot(data)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training loss', 'Validation loss'])
    plt.savefig(data_dir+'performance.jpg')