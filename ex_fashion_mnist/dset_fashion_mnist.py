from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Training settings
def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Fashion-MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    return parser.parse_args("")


# load data
def load_data(train_batch_size,
              test_batch_size,
              device,
              data_dir='data',
              shuffle=True,
              shuffle_target=False,
              return_indices=False):
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
    FashionMNIST_dataset = datasets.FashionMNIST
    if return_indices:
        FashionMNIST_dataset = dataset_with_indices(FashionMNIST_dataset)
        
    train_set = FashionMNIST_dataset(data_dir, train=True, download=True,
                   transform=transforms.Compose([transforms.ToTensor()]))
    test_set = FashionMNIST_dataset(data_dir, train=False, download=True,
                   transform=transforms.Compose([transforms.ToTensor()]))
    
    if shuffle_target:
        train_set.targets[:30000] = torch.randint(0, 10, (30000,))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=shuffle, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader


def train(train_loader, test_loader, model, args):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda == "True":
                data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
        train_loss, train_acc = test(train_loader, model, args, verbose=False)
        test_loss, test_acc = test(test_loader, model, args, verbose=False)
        print('====> Epoch: {} Average train loss: {:.4f}, Average test loss: {:.4f} (accuracies: {:.2f}%, {:.2f}%)'.format(epoch, train_loss, test_loss, train_acc, test_acc))
        
        if epoch + 1 in [5, 10, 20, 30, 40, 50, 70, 100, 200, 300, 500]:
            torch.save(model.state_dict(), 'models/FashionCNN_ShuffleLabel50%_epochs={}.pth'.format(epoch+1))
        
    return model


def test(test_loader, model, args, verbose=True):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda == "True":
            data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').data.item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    if verbose:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    else:
        return test_loss, 100.*correct/len(test_loader.dataset)


def pred_ims(model, ims, layer='softmax', device='cuda'):
    if len(ims.shape) == 2:
        ims = np.expand_dims(ims, 0)
    ims_torch = torch.unsqueeze(torch.Tensor(ims), 1).float().to(device) # cuda()
    preds = model(ims_torch)

    # todo - build in logit support
    # logits = model.logits(t)
    return preds.data.cpu().numpy()




# mnist dataset with return index
# dataset
def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """
    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })


if __name__ == '__main__':
    from model import Net
    args = get_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    train_loader, test_loader = load_data(args.batch_size, args.test_batch_size, args.cuda)

    # create model
    model = Net()
    if args.cuda:
        model.cuda()

    # train
    for epoch in range(1, args.epochs + 1):
        model = train(epoch, train_loader)
        test(model, test_loader)

    # save
    torch.save(model.state_dict(), 'mnist.model')
    # load and test
    # model_loaded = Net().cuda()
    # model_loaded.load_state_dict(torch.load('mnist.model'))
    # test(model_loaded, test_loader)
