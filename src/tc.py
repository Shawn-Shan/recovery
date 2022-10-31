import numpy as np
import torch
import torchvision
from sklearn.model_selection import train_test_split
from termcolor import colored
from torch.cuda.amp import GradScaler
from torch.optim import SGD, lr_scheduler
from torch.utils.data import Dataset
import pickle
from keras.datasets import cifar10

ts = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                     torchvision.transforms.RandomCrop(32, padding=4),
                                     torchvision.transforms.RandomHorizontalFlip(),
                                     ])

test_ts = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])


def cpu(tensor):
    if type(tensor) is np.ndarray:
        return tensor

    array = tensor.detach().cpu().numpy()
    return array


def gpu(array):
    tp_1 = type(array)
    if tp_1 == list:
        array = np.array(array)
        tp_1 = type(array)

    if tp_1 != np.ndarray:
        tp = array.type()
        if "cuda" in tp:
            return array
        else:
            return array.cuda()

    array = np.array(array)
    array_type = str(array.dtype)
    tensor = torch.tensor(array).cuda()
    if array_type.startswith("float") or array_type.startswith("double"):
        tensor = tensor.type(torch.cuda.FloatTensor)
    return tensor


def switch(array):
    return switch_order(array)


def switch_order(array):
    if not isinstance(array, np.ndarray):
        array = cpu(array)

    shape = array.shape
    if len(shape) == 4:
        if shape[0] == 1:
            shape = shape[1:]
            array = array[0]
        else:
            raise Exception("Do not support 4D batch")

    if (shape[0] == 3 or shape[0] == 1) and shape[1] == shape[2]:
        c_first = True
    else:
        c_first = False
    if c_first:
        array = array.transpose((1, 2, 0))
    else:
        array = array.transpose((2, 0, 1))
    return array


def transform(x, ts=test_ts, dim=False):
    x = np.array(x, dtype=np.float32)
    if len(x.shape) == 4:
        x = [ts(i) for i in x]
        x = torch.stack(x)
    else:
        x = ts(x)
    x = x.cuda()
    if dim and len(x.shape) == 3:
        x = x[None, :]
    return x


def load_dataset(dataset):
    if dataset == 'cifar10':
        (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
        X_train = X_train / 255.0
        X_test = X_test / 255.0
    else:
        raise Exception("Dataset not implemented")

    return X_train, Y_train, X_test, Y_test


class CIFARDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = np.array(X, dtype=np.float32)
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        cur_x = self.X[idx]
        cur_y = self.y[idx]
        cur_x = transform(cur_x, self.transform)
        return cur_x, cur_y


def get_loader(X, Y, ts, batch_size=512):
    dataset = CIFARDataset(X, Y, ts)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def get_opt(model, train_size, lr=0.5, batch_size=512, epochs=24):
    opt = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    iters_per_epoch = train_size // batch_size
    lr_schedule = np.interp(np.arange((epochs + 1) * iters_per_epoch),
                            [0, 5 * iters_per_epoch, epochs * iters_per_epoch],
                            [0, 1, 0])

    scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
    scaler = GradScaler()
    return opt, scheduler, scaler


def load_resnet18():
    import resnet
    return resnet.ResNet18()


def load_model(model_path=None, arch="resnet18"):
    if arch == 'resnet18':
        model = load_resnet18().cuda()
    else:
        raise Exception("Unimplemented")
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def pickle_write(data, outfile):
    return pickle.dump(data, open(outfile, "wb"))


def pickle_read(infile):
    return pickle.load(open(infile, "rb"))


def pgd_torch(models, cur_X, cur_labels, targeted=False, iters=10, alpha=0.01, eps=0.05):
    images = cur_X
    loss = torch.nn.CrossEntropyLoss()
    for m in models:
        m.eval()

    ori_images = images.data

    for i in range(iters):
        images.requires_grad = True
        cost = 0
        for m in models:
            outputs = m(images)
            m.zero_grad()
            cost += loss(outputs, cur_labels).cuda()

        cost.backward()
        if not targeted:
            adv_images = images + alpha * images.grad.sign()
        else:
            adv_images = images - alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)

        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    return images


def get_sub_data(rd, loader=True, test=True, n_sample=512):
    all_x = []
    all_y = []

    for label, file in enumerate(rd):
        cur_X = pickle_read(file)
        cur_X = cur_X[:n_sample]
        label = label % 10
        all_x.append(cur_X)
        all_y.append(np.array([label] * len(cur_X)))

    all_x_np = np.concatenate(all_x)
    all_y_np = np.concatenate(all_y)

    if test:
        X_train_s, X_test_s, Y_train_s, Y_test_s = train_test_split(all_x_np, all_y_np, test_size=0.05)

    if loader:
        train_loader_s = get_loader(all_x_np, all_y_np, ts)
        if test:
            test_loader_s = get_loader(X_test_s, Y_test_s, test_ts)
        else:
            test_loader_s = None
        return train_loader_s, test_loader_s

    return transform(all_x_np), gpu(all_y_np)


def get_target_data(X, Y, y_target, source_only=True):
    if len(Y.shape) > 1:
        Y = np.argmax(Y, axis=1)
    select_list = Y == y_target
    X_select = X[select_list]
    Y_select = Y[select_list]
    if source_only:
        return X_select, Y_select
    unselect_list = Y != y_target
    X_unselect = X[unselect_list]
    Y_unselect = Y[unselect_list]
    return X_select, Y_select, X_unselect, Y_unselect
