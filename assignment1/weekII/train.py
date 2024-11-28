from torchvision import datasets, transforms
import torch


def get_train_loader(batch_size, **kwargs):
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)),
                       lambda x: x>0,
                       lambda x: x.float(),
                        ])),
                        batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader


def get_test_loader(test_batch_size, **kwargs):
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)),
                       lambda x: x>0,
                       lambda x: x.float(),
                        ])),
                        batch_size=test_batch_size, shuffle=True, **kwargs)
    return test_loader