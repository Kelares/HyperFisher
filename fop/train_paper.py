import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
from util.cutout import Cutout
from model.resnet import ResNet18
import os
        
dataset_options = ['cifar10', 'cifar100']

parser = argparse.ArgumentParser(description='CIFAR10')
parser.add_argument('--dataset', '-d', default='cifar10',
                    choices=dataset_options)
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--learning_rate', type=float, default=0.1,
                    help='learning rate')
parser.add_argument('--data_augmentation', action='store_true', default=False,
                    help='augment data by flipping and cropping')
parser.add_argument('--cutout', action='store_true', default=False,
                    help='apply cutout')
parser.add_argument('--n_holes', type=int, default=1,
                    help='number of holes to cut out from image')
parser.add_argument('--length', type=int, default=16,
                    help='length of the holes')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 1)')
parser.add_argument('--clipping_norm', type=float, default=10,
                    help='gradient clipping norm (default: -1, no clipping)')
parser.add_argument('--optim', type=str, default='sgd',
                    help='optimizer type (default: sgd)')
parser.add_argument('--damping', type=float, default=0.01,
                    help='damping for curvature estimation (default: 0.01)')
parser.add_argument('--curvature_update_interval', type=int, default=100,
                    help='interval for curvature update (default: 100)')
parser.add_argument('--ema_decay', type=float, default=0.05,
                    help='exponential moving average decay for curvature (default: 0.05)')
parser.add_argument('--beta_adaptive', action='store_true', default=False,
                    help='use adaptive beta for curvature estimation')
parser.add_argument('--eta_adaptive', action='store_true', default=False,
                    help='use adaptive eta for curvature estimation')
parser.add_argument('--beta', type=float, default=0.01,
                    help='beta value for curvature estimation (default: 0.01)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
cudnn.benchmark = True

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

def setup_ddp():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    print(f"[Rank {rank}] using GPU {local_rank}")
    return local_rank, rank, world_size

args.distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
if args.distributed:
    args.local_rank, args.rank, args.world_size = setup_ddp()
else:
    args.local_rank = 0
    args.rank = 0
    args.world_size = 1

print(args)

normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

train_transform = transforms.Compose([])
if args.data_augmentation:
    train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
    train_transform.transforms.append(transforms.RandomHorizontalFlip())
train_transform.transforms.append(transforms.ToTensor())
train_transform.transforms.append(normalize)
if args.cutout:
    train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))

test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize])

if args.dataset == 'cifar10':
    num_classes = 10
    train_dataset = datasets.CIFAR10(root='/scratch/',
                                     train=True,
                                     transform=train_transform,
                                     download=True)

    test_dataset = datasets.CIFAR10(root='/scratch/',

                                    train=False,
                                    transform=test_transform,
                                    download=True)
elif args.dataset == 'cifar100':
    num_classes = 100
    train_dataset = datasets.CIFAR100(root='/tmp/',
                                      train=True,
                                      transform=train_transform,
                                      download=True)

    test_dataset = datasets.CIFAR100(root='/tmp/',
                                     train=False,
                                     transform=test_transform,
                                     download=True)

if args.distributed:
    batch_size_per_gpu = max(1, args.batch_size // args.world_size)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size_per_gpu,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=2,
    )
else:
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=2)


model = ResNet18(num_classes=num_classes)

device = torch.device(f"cuda:{args.local_rank}" if args.cuda else "cpu")
model = model.to(device)


criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

if args.optim == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                    momentum=0.9, nesterov=True, weight_decay=5e-4)
elif args.optim == 'adamw':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=5e-4)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                    momentum=0.9, nesterov=False, weight_decay=5e-4)
    import asdl
    from fop  import FOPGradientMaker,DistKFACGradientMaker
    config = asdl.PreconditioningConfig(data_size=args.batch_size,
                                damping=args.damping,
                                ema_decay = args.ema_decay,
                                preconditioner_upd_interval=args.curvature_update_interval,
                                curvature_upd_interval=args.curvature_update_interval,
                                ignore_modules=[nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d,nn.LayerNorm]
                                )
    if "kfac" in args.optim:
        grad_maker = DistKFACGradientMaker(model=model, config=config)
    elif "fop" in args.optim:
        grad_maker = FOPGradientMaker(model, config,beta_adaptive=args.beta_adaptive,eta_adaptive=args.eta_adaptive,beta=args.beta)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)


def test(loader):
    model.eval()
    correct = 0.
    total = 0.
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            pred = model(images)

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    val_acc = correct / total
    model.train()
    return val_acc


print(f"len of loader: {len(train_loader)}")

for epoch in range(args.epochs):
    if args.distributed:
        train_sampler.set_epoch(epoch)
    if args.cuda:
        torch.cuda.reset_peak_memory_stats()

    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.

    profiled = False

    progress_bar = tqdm(train_loader)
    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        model.zero_grad()

        def train_step():
            if "fop" in args.optim or 'kfac' in args.optim:
                if dist.is_initialized():
                    return grad_maker.multi_run(
                    model, images, labels, torch.nn.functional.cross_entropy,
                    label_smoothing=0.1)
                else:
                    return grad_maker.run(
                        model, images, labels, torch.nn.functional.cross_entropy,
                        label_smoothing=0.1)
            else:
                pred = model(images)
                xentropy_loss = criterion(pred, labels)
                xentropy_loss.backward()
                return pred, xentropy_loss


        pred, xentropy_loss = train_step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipping_norm) if args.clipping_norm != -1 else None
        optimizer.step()
        try:
            xentropy_loss_avg += xentropy_loss.item()
        except:
            xentropy_loss_avg += xentropy_loss


        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total



    test_acc = test(test_loader)
    if args.local_rank == 0:

        tqdm.write('test_acc: %.3f' % (test_acc))


        max_mem = torch.cuda.max_memory_allocated() / (1024 ** 3) 
        print(f"Max GPU Memory Allocated: {max_mem:.2f} MB")
        max_mem_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
        print(f"Max GPU Memory Reserved: {max_mem_reserved:.2f} GB")

        print(f'[RANK {args.local_rank}] test_acc: {test_acc:.3f}, lr: {optimizer.param_groups[0]["lr"]}', flush=True)

        

    scheduler.step(test_acc)


if args.distributed:
    dist.destroy_process_group()  
    print(f"[RANK {args.local_rank}] Distributed training complete.")
else:
    print("Non-distributed training complete.")
