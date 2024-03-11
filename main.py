# Copyright information
# Author: Ye Xu, Ya Gao, Xiaorong Qiu, Yang Chen, Ying Ji
# Affiliation: School of IoT Engineering, Wuxi Institute of Technology

import argparse, os

parser = argparse.ArgumentParser(description='reading the parameters required for training the model.')

parser.add_argument('-mn', '--model_name', type=str, default='resnet18', help='Model Name (default: resnet18, options：resnet18, resnet34, resnet50, resnet101, mobilenet, tiny-swin)')
parser.add_argument('-ds', '--dataset_name', type=str, default='cifar100', help='Dataset Name (default：cifar100, options: food101, miniimagenet, oxfordiiipet, caltech256)')
parser.add_argument('-pr', '--pretrained', type=bool, default=False, help='Using a Pre-trained Model or Not (default: False)')
parser.add_argument('-bs', '--batch_size', type=int, default=128, help='Batch Size (default: 128)')
parser.add_argument('-alpha', '--alpha', type=float, default=1, help='Alpha (default: 1)')
parser.add_argument('-beta', '--beta', type=float, default=0, help='Beta (default: 0)')
parser.add_argument('-itrm', '--inter_mixup', type=str, default='none', help='Mixup Method (default: None，options: Mixup, Manifold_Mixup)')
parser.add_argument('-rd', '--root_dir', type=str, default='./data', help='Root Path (default: ./data)')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.1, help='Learning Rate (default: 0.1)')
parser.add_argument('-mo', '--momentum', type=float, default=0.9, help='Momentum (default: 0.9)')
parser.add_argument('-wd', '--weight_decay', type=float, default=5e-4, help='Weight Decay (default: 5e-4)')
parser.add_argument('-ss', '--step_size', type=int, default=10, help='Step Size for Learning Rate Decay (default: 10)')
parser.add_argument('-ga', '--gamma', type=float, default=0.5, help='Gamma for Learning Rate Decay (default: 0.5)')
parser.add_argument('-seed', '--seed', type=int, default=123, help='Random Seed (default: 123)')
parser.add_argument('-ep', '--epochs', type=int, default=120, help='Total Number of Epochs (default: 120)')
parser.add_argument('-ne', '--nesterov', type=bool, default=False, help='Using Nesterov or Not (default: False)')
parser.add_argument('-phase', '--phase', type=str, default='test', help='Testing or Validation (default：test, options: test, val)')
parser.add_argument('-vr', '--val_ratio', type=float, default=0.1, help='Validation Ratio (default：0.1)')
parser.add_argument('-mp', '--mixed_precision', type=bool, default=False, help='Using Mixed Precision Training or Not (default：False)')

args = parser.parse_args()

#Setting random seed
import torch
import numpy as np, random
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

set_seed(args.seed)

#Defining methods for intra-class mixup and inter-class mixup with incidental intra-class mixup.
def inter_mixup_inc(x, y, lam, use_cuda=True):
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def intra_mixup(features, labels):
    syn_features, syn_labels = [], []
    unique_labels = labels.unique()

    for label in unique_labels:
        mask = labels == label
        class_specific_features = features[mask]
        if len(class_specific_features) > 0:
            weights = torch.rand(len(class_specific_features), device=features.device)
            weights /= weights.sum()
            syn_feature = (class_specific_features * weights.unsqueeze(-1)).sum(dim=0)
            syn_features.append(syn_feature)
            syn_labels.append(label.item())

    syn_features = torch.stack(syn_features)
    syn_labels = torch.tensor(syn_labels, dtype=torch.long, device=features.device)
    return syn_features, syn_labels

# Customizing the dataset to return the original image, augmented image, and label.
import torchvision
from PIL import Image
if args.dataset_name == 'cifar100':
    class CustomCIFAR100(torchvision.datasets.CIFAR100):
        def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
            super().__init__(root, train=train, transform=transform, target_transform=target_transform,
                             download=download)
            self.original_transform = transforms.Compose([transforms.ToTensor(),
                                                          transforms.Normalize((0.485, 0.456, 0.406),
                                                                               (0.229, 0.224, 0.225))])
            self._labels = self.targets
        def __getitem__(self, index):
            img, target = self.data[index], self.targets[index]
            original_img = self.original_transform(img.copy())
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return original_img, img, target
elif args.dataset_name == 'food101':
    class CustomFood101(torchvision.datasets.Food101):
        def __init__(self, root, split, transform=None, target_transform=None, download=False):
            super().__init__(root, split=split, transform=transform, target_transform=target_transform,
                             download=download)
            self.original_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize((0.485, 0.456, 0.406),
                                                                               (0.229, 0.224, 0.225))])
            self.classes = np.unique(self._labels)

        def __getitem__(self, index):
            img_path, target = self._image_files[index], self._labels[index]
            img = Image.open(img_path).convert('RGB')
            original_img = self.original_transform(img.copy())
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)

            return original_img, img, target
elif args.dataset_name == 'miniimagenet':
    from MLclf import MLclf
    class CustomMiniImagenet:
        def __init__(self, split=None, transform=None, target_transform=None, download=False):
            MLclf.miniimagenet_download(Download=download)
            self.train_dataset, self.validation_dataset, self.test_dataset = MLclf.miniimagenet_clf_dataset(
                ratio_train=0.8, ratio_val=0.0,
                seed_value=100, shuffle=True,
                transform=None,
                save_clf_data=True)
            if split == 'train':
                self.data = self.train_dataset.tensors[0]
                self._labels = self.train_dataset.tensors[1]
            else:
                self.data = self.test_dataset.tensors[0]
                self._labels = self.test_dataset.tensors[1]

            self.classes = np.unique(self._labels)
            self.transform = transform
            self.target_transform = target_transform
            self.original_transform = transforms.Compose([transforms.ToTensor(),
                                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        def __getitem__(self, index):
            img, target = self.data[index], self._labels[index]
            img = img.permute((1, 2, 0))
            img = Image.fromarray((img.numpy() * 255).astype('uint8'))
            original_img = self.original_transform(img.copy())
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)

            return original_img, img, target

        def __len__(self) -> int:
            return len(self._labels)
elif args.dataset_name == 'caltech256':
    class CustomCaltech256(torchvision.datasets.Caltech256):
        def __init__(self, root, split=None, transform=None, target_transform=None, download=False):
            super().__init__(root, transform=transform, target_transform=target_transform, download=download)
            self.original_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize((0.485, 0.456, 0.406),
                                                                               (0.229, 0.224, 0.225))])

            self.prepare_split()

            if split == 'train':
                self.data = self.train_data
                self.targets = self.train_targets
            else:
                self.data = self.test_data
                self.targets = self.test_targets

            self._labels = self.targets
            self.classes = np.unique(self._labels)

        def prepare_split(self):
            self.y = np.array(self.y)
            self.index = np.array(self.index)

            train_data = []
            train_targets = []
            test_data = []
            test_targets = []

            for c in range(256 + 1):
                class_data = self.index[np.where(self.y == c)[0]]
                class_targets = self.y[np.where(self.y == c)[0]]

                for i, cd in enumerate(class_data):

                    if i < 60:
                        train_data.append(os.path.join(
                            self.root,
                            "256_ObjectCategories",
                            self.categories[c],
                            f"{c + 1:03d}_{cd:04d}.jpg",
                        ))
                    elif i >= 60 and i < 80:
                        test_data.append(os.path.join(
                            self.root,
                            "256_ObjectCategories",
                            self.categories[c],
                            f"{c + 1:03d}_{cd:04d}.jpg",
                        ))

                train_targets.extend(class_targets[:60])
                test_targets.extend(class_targets[60:80])

            self.train_data = train_data
            self.train_targets = train_targets
            self.test_data = test_data
            self.test_targets = test_targets

        def __getitem__(self, index):
            img_path, target = self.data[index], self.targets[index]
            img = Image.open(img_path).convert('RGB')
            original_img = self.original_transform(img.copy())
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return original_img, img, target

        def __len__(self) -> int:
            return len(self.targets)
elif args.dataset_name == 'oxfordiiipet':
    class CustomOxfordIIITPet(torchvision.datasets.OxfordIIITPet):
        def __init__(self, root, split, transform=None, target_transform=None, download=False):
            super().__init__(root, split=split, transform=transform, target_transform=target_transform,
                             download=download)
            self.original_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize((0.485, 0.456, 0.406),
                                                                               (0.229, 0.224, 0.225))])
            self.classes = np.unique(self._labels)

        def __getitem__(self, index):
            img_path, target = self._images[index], self._labels[index]
            img = Image.open(img_path).convert('RGB')
            original_img = self.original_transform(img.copy())
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)

            return original_img, img, target

# Creating a custom batch sampler to ensure that each class has at least 2 image samples in each batch.
from torch.utils.data import BatchSampler

class BalancedBatchSampler(BatchSampler):
    def __init__(self, labels, batch_size):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.unique_labels = np.unique(labels)
        self.list_indices = [np.where(labels == label)[0] for label in self.unique_labels]

    def __iter__(self):
        random_idx = np.arange(len(self.labels))
        np.random.shuffle(random_idx)
        random_labels = self.labels[random_idx]

        for i in range(np.ceil(len(self.labels) // self.batch_size).astype(int)):
            batch_idx = random_idx[i * self.batch_size : np.min(((i + 1) * self.batch_size, len(random_idx)))]
            batch_labels = random_labels[i * self.batch_size : np.min(((i + 1) * self.batch_size, len(random_idx)))]

            for c, label in enumerate(self.unique_labels):
                if np.sum(batch_labels == label) == 1:
                    repeat_index = -1
                    while repeat_index in batch_idx or repeat_index == -1:
                        repeat_index = np.random.choice(self.list_indices[c])
                    batch_idx = np.concatenate((batch_idx, [repeat_index]))

            yield batch_idx

    def __len__(self):
        return np.ceil(len(self.labels) // self.batch_size).astype(int)

# Specifing the data augmentation strategy.
import torchvision.transforms as transforms
from utils.cutout import Cutout
if args.dataset_name == 'cifar100':
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor()
            , transforms.RandomCrop(32, padding=4)  # 先四周填充0，在吧图像随机裁剪成32*32
            , transforms.RandomHorizontalFlip(p=0.5)  # 随机水平翻转 选择一个概率概率
            , Cutout(n_holes=1, length=16)
            , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
        ]),
        'test': transforms.Compose([
            transforms.ToTensor()
            , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
elif args.dataset_name == 'miniimagenet':
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor()
            , transforms.RandomCrop(84, padding=12)
            , transforms.RandomHorizontalFlip(p=0.5)
            , Cutout(n_holes=1, length=32),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
    }
elif args.dataset_name == 'food101':
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    }
elif args.dataset_name == 'oxfordiiipet':
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    }
elif args.dataset_name == 'caltech256':
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    }

# Setting up the training and testing datasets.
from torch.utils.data import DataLoader
if args.dataset_name == 'cifar100':
    train_dataset = CustomCIFAR100(root=args.root_dir, train=True, download=True, transform=data_transforms['train'])
    test_dataset = CustomCIFAR100(root=args.root_dir, train=False, download=True, transform=data_transforms['test'])
elif args.dataset_name == 'food101':
    train_dataset = CustomFood101(root=args.root_dir, split='train', download=True, transform=data_transforms['train'])
    test_dataset = CustomFood101(root=args.root_dir, split='test', download=True, transform=data_transforms['test'])
elif args.dataset_name == 'miniimagenet':
    train_dataset = CustomMiniImagenet(split='train', download=False, transform=data_transforms['train'])
    test_dataset = CustomMiniImagenet(split='test', download=False, transform=data_transforms['test'])
elif args.dataset_name == 'caltech256':
    train_dataset = CustomCaltech256(root=args.root_dir, split='train', download=True, transform=data_transforms['train'])
    test_dataset = CustomCaltech256(root=args.root_dir, split='test', download=True, transform=data_transforms['test'])
elif args.dataset_name == 'oxfordiiipet':
    train_dataset = CustomOxfordIIITPet(root=args.root_dir, split='trainval', download=True, transform=data_transforms['train'])
    test_dataset = CustomOxfordIIITPet(root=args.root_dir, split='test', download=True, transform=data_transforms['test'])

#If the current phase is validation, split a specified proportion of the data from the training set to create
#a validation set, using the remaining portion as the training set. Designate the validation set as the testing set.
if args.phase == 'val':
    from torch.utils.data import Subset
    from sklearn.model_selection import StratifiedShuffleSplit

    def stratified_random_split(dataset, val_split_ratio):
        labels = dataset.targets if hasattr(dataset, 'targets') else dataset._labels
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split_ratio, random_state=42)
        indices = list(range(len(dataset)))
        for train_index, val_index in sss.split(indices, labels):
            train_subset = Subset(dataset, train_index)
            val_subset = Subset(dataset, val_index)

        return train_subset, val_subset

    train_dataset, test_dataset = stratified_random_split(train_dataset, args.val_ratio)
    train_dataset._labels = np.array(train_dataset.dataset._labels)[train_dataset.indices]
    train_dataset.classes = train_dataset.dataset.classes


trainloader = DataLoader(train_dataset, num_workers=12, pin_memory=True, batch_sampler=BalancedBatchSampler(train_dataset._labels, args.batch_size))
testloader = DataLoader(test_dataset, batch_size=48, shuffle=False, num_workers=12)

#Building the model and modifing the final classification layer.
if args.model_name == 'resnet18':
    model = torchvision.models.resnet18(pretrained=args.pretrained)
elif args.model_name == 'resnet34':
    model = torchvision.models.resnet34(pretrained=args.pretrained)
elif args.model_name == 'resnet50':
    model = torchvision.models.resnet50(pretrained=args.pretrained)
elif args.model_name == 'resnet101':
    model = torchvision.models.resnet101(pretrained=args.pretrained)
elif args.model_name == 'tiny-swin':
    model = torchvision.models.swin_t(pretrained=args.pretrained)
elif args.model_name == 'mobilenet':
    model = torchvision.models.mobilenet_v3_large(pretrained=args.pretrained)

import torch.nn as nn
if args.dataset_name in ['cifar100', 'miniimagenet']:
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    model.maxpool = nn.MaxPool2d(1, 1, 0)

if args.model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101']:
    model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
elif args.model_name in ['tiny-swin']:
    model.head = nn.Linear(model.head.in_features, len(train_dataset.classes))
elif args.model_name in ['mobilenet']:
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(train_dataset.classes))

#Implementing Mixup and Manifold Mixup.
if args.model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101']:
    def forward(self, x, target= None, mixup=False, mixup_hidden=False, mixup_alpha=None, only_features=False):
        if mixup_hidden:
            layer_mix = random.randint(0, 2)
        elif mixup:
            layer_mix = 0
        else:
            layer_mix = None

        out = x
        if mixup or mixup_hidden:
            lam = np.random.beta(mixup_alpha, mixup_alpha)

        if layer_mix == 0:
            out, y_a, y_b, lam = inter_mixup_inc(out, target, lam=lam)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        if layer_mix == 1:
            out, y_a, y_b, lam = inter_mixup_inc(out, target, lam=lam)
        out = self.layer2(out)
        if layer_mix == 2:
            out, y_a, y_b, lam = inter_mixup_inc(out, target, lam=lam)
        out = self.layer3(out)
        if layer_mix == 3:
            out, y_a, y_b, lam = inter_mixup_inc(out, target, lam=lam)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        if only_features:
            return out
        else:
            out = self.fc(out)
            if mixup or mixup_hidden:
                return out, y_a, y_b, lam
            else:
                return out

    torchvision.models.ResNet.forward = forward
elif args.model_name in ['tiny-swin']:
    def forward(self, x, target= None, mixup=False, mixup_hidden=False, mixup_alpha=None, only_features=False):
        if mixup_hidden:
            layer_mix = random.randint(0, 3)
        elif mixup:
            layer_mix = 0
        else:
            layer_mix = None

        out = x
        if mixup or mixup_hidden:
            lam = np.random.beta(mixup_alpha, mixup_alpha)

        if layer_mix == 0:
            out, y_a, y_b, lam = inter_mixup_inc(out, target, lam=lam)

        for i, fea_module in enumerate(self.features):
            if (layer_mix == 1 and i == 0) or (layer_mix == 2 and i == 1) or (layer_mix == 3 and i == 3) or (layer_mix == 4 and i == 5):
                out, y_a, y_b, lam = inter_mixup_inc(out, target, lam=lam)
            out = fea_module(out)
        out = self.norm(out)
        out = self.permute(out)
        out = self.avgpool(out)
        out = self.flatten(out)

        if only_features:
            return out
        else:
            out = self.head(out)
            if mixup or mixup_hidden:
                return out, y_a, y_b, lam
            else:
                return out

    torchvision.models.SwinTransformer.forward = forward
elif args.model_name in ['mobilenet']:
    def forward(self, x, target= None, mixup=False, mixup_hidden=False, mixup_alpha=None, only_features=False):
        if mixup_hidden:
            layer_mix = random.randint(0, 3)
        elif mixup:
            layer_mix = 0
        else:
            layer_mix = None

        out = x
        if mixup or mixup_hidden:
            lam = np.random.beta(mixup_alpha, mixup_alpha)

        if layer_mix == 0:
            out, y_a, y_b, lam = inter_mixup_inc(out, target, lam=lam)

        for i, fea_module in enumerate(self.features):
            if (layer_mix == 1 and i == 0) or (layer_mix == 2 and i == 1) or (layer_mix == 3 and i == 2):
                out, y_a, y_b, lam = inter_mixup_inc(out, target, lam=lam)
            out = fea_module(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        out = self.classifier[0](out)
        out = self.classifier[1](out)
        out = self.classifier[2](out)

        if only_features:
            return out
        else:
            out = self.classifier[3](out)
            if mixup or mixup_hidden:
                return out, y_a, y_b, lam
            else:
                return out

    torchvision.models.MobileNetV3.forward = forward


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Creating the optimizer and scheduler.
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

# Generating a log file to track training and testing progress.
import datetime, time
os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)

current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_file = os.path.join('logs', f'log_{current_time}.txt')

with open(log_file, 'a') as f:
    for arg in vars(args):
        f.write(f"{arg}: {getattr(args, arg)}\n")
        print(f"{arg}: {getattr(args, arg)}")
    f.write('-'*80 + '\n')

#Training and testing the model.
if args.mixed_precision:
    from torch.cuda.amp import GradScaler, autocast
    scaler = GradScaler()
else:
    class autocast:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_value, traceback):
            return False

for epoch in range(args.epochs):  # loop over the dataset multiple times
    start_time = time.time()
    model.train()
    train_loss = 0.0

    for i, (ori_inputs, aug_inputs, labels) in enumerate(trainloader, 0):
        ori_inputs, aug_inputs, labels = ori_inputs.cuda(), aug_inputs.cuda(), labels.cuda()

        with autocast():
            loss1 = 0
            if args.beta > 0: #If beta is greater than 0, perform intra-class mixup.
                ori_features = model(ori_inputs, only_features=True)
                syn_features, syn_labels = intra_mixup(ori_features, labels)
                if args.model_name == 'tiny-swin':
                    syn_out = model.head(syn_features)
                elif args.model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101']:
                    syn_out = model.fc(syn_features)
                elif args.model_name == 'mobilenet':
                    syn_out = model.classifier[3](syn_features)
                loss1 = criterion(syn_out, syn_labels)

            #If args.inter_mixup is 'Mixup', apply MixUp. If it is 'Manifold_Mixup', apply Manifold MixUp.
            out = model(aug_inputs, labels, mixup=args.inter_mixup == 'Mixup',
                           mixup_hidden=args.inter_mixup == 'Manifold_Mixup', mixup_alpha=args.alpha)
            if args.inter_mixup in ['Mixup', 'Manifold_Mixup']:
                out, y_a, y_b, lam = out
                loss2 = mixup_criterion(criterion, out, y_a, y_b, lam)
            else:
                loss2 = criterion(out, labels)

            total_loss = args.beta * loss1 + (1 - args.beta) * loss2

        optimizer.zero_grad()
        if args.mixed_precision:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        train_loss += loss2.item()

    avg_train_loss = train_loss / (i + 1)
    end_time = time.time()
    epoch_duration = end_time - start_time

    with open(log_file, 'a') as f:
        f.write(f'[{epoch + 1}, {i + 1}] train loss: {avg_train_loss:.10f}, learning_rate: {scheduler.get_last_lr()[0]}, training time: {epoch_duration:.2f} seconds\n')
        print(f'[{epoch + 1}, {i + 1}] train loss: {avg_train_loss:.10f}, learning_rate: {scheduler.get_last_lr()[0]}, training time: {epoch_duration:.2f} seconds')

    scheduler.step()

    model.eval()
    test_loss = 0
    with torch.no_grad():
        correct = 0
        total = 0
        for j, (ori_images, _, labels) in enumerate(testloader):
            labels = labels.cuda()
            ori_images = ori_images.cuda()
            outputs = model(ori_images)

            loss1 = criterion(outputs, labels)
            test_loss += loss1.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        with open(log_file, 'a') as f:
            f.write(f'[{epoch + 1}, {j + 1}] test loss: {test_loss / (j + 1):.10f}, Accuracy: {100 * correct / total} %\n')
            print(f'[{epoch + 1}, {j + 1}] test loss: {test_loss / (j + 1):.10f}, Accuracy: {100 * correct / total} %')

model_path = os.path.join('models', f'model_{current_time}.pth')
torch.save(model.state_dict(), model_path)
print('Finished')
