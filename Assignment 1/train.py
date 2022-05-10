import torch
import argparse
import torch.nn.functional as F
import numpy as np

from model import LinearNet
import MNIST_dataloader
import result_plot


def get_args():
    parser = argparse.ArgumentParser('training parameters')
    parser.add_argument('--bs', type=int, default=32)             #batch size
    parser.add_argument('--dl', type=str, default='Datasets')      #data loc
    parser.add_argument('--lr', type=float, default=0.01)          #learning rate
    parser.add_argument('--EPOCH', type=int, default=150)           #epoch num
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--save_file', type=str, default='./log/') #saving path
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    return args

def split_train_val_test():
    args = get_args()
    train_loader, test_loader = MNIST_dataloader.create_dataloaders(args.dl, args.bs)
    train_db, val_db = torch.utils.data.random_split(train_loader, [1500, 375])
    return train_db.dataset, val_db.dataset, test_loader

def prediction(model, test_loader, model_id, device):
    args = get_args()
    img_in = enumerate(test_loader)
    _, (clean_, noise_, _) = next(img_in)
    noise_in = noise_.view(args.bs,-1).to(device)
    output = model(noise_in)
    output = output.detach().numpy()
    pred = output.reshape(args.bs, 1, 32, 32)
    result_plot.plot_img(clean_, noise_, pred, model_id)

def train(args, model, device, train_set, optimizer, loss_function):
    model.train()
    train_loss = 0
    count = 0
    for _, (clean_, noise_, _) in enumerate(train_set):
        noise_, clean_ = noise_.view(args.bs,-1).to(device), clean_.view(args.bs,-1).to(device)
        optimizer.zero_grad()
        output = model(noise_)
        loss = loss_function(output, clean_)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        count += 1
    return train_loss/count

def evaluation(args, model, device, test_loader, loss_function):
    model.eval()
    count = 0
    test_loss = 0
    with torch.no_grad():
        for _, (clean_, noise_, _) in enumerate(test_loader):
            noise_, clean_ = noise_.view(args.bs,-1).to(device), clean_.view(args.bs,-1).to(device)
            output = model(noise_)
            loss = loss_function(output, clean_)
            test_loss += loss.item()
            count += 1
    return test_loss/count

def main():
    args = get_args()
    device = torch.device("cuda" if args.use_cuda else "cpu")
    train_set, val_set, test_set = split_train_val_test()

    # training parameters
    model = LinearNet(32*32, 200, 200).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr) 
    loss_function = torch.nn.MSELoss()
    
    #initial test
    state = {'net':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'epoch':0}
    torch.save(state, args.save_file + 'initialModel.pth')
    model_id = 'initial'
    prediction(model, test_set, model_id, device)
    # training and evaluation
    print('training start')
    train_loss_list = []
    val_loss_list = []
    for epoch in range(args.EPOCH):
        train_loss = train(args, model, device, train_set, optimizer, loss_function)
        val_loss = evaluation(args, model, device, val_set, loss_function)
        print(f'\nepoch:{epoch} \n train_loss:{train_loss} \t val loss:{val_loss}')
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

    #last test
    state = {'net':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'epoch':epoch}
    torch.save(state, args.save_file + 'lastModel.pth')
    model_id = 'last'
    prediction(model, test_set, model_id, device)
            
    np.savetxt(args.save_file + 'train_loss_list.txt', np.array(train_loss_list))
    np.savetxt(args.save_file + 'val_loss_list.txt', np.array(val_loss_list))

    result_plot.plot_save(result_plot.variable_name, result_plot.data_dir)

if __name__ == '__main__':
    main()