import argparse
import logging
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18
from torchvision.transforms import Compose, Lambda, Normalize, Resize
from tqdm import tqdm

sys.path.append(os.path.abspath(os.curdir))

from chronica.torch.utils.data import DistributedSampler
from tests.datasets import HMDB51


def run(epochs: int, lr: float, root: str):
    dist.init_process_group(backend=dist.Backend.NCCL)

    # environment variable set by torchrun
    rank = int(os.getenv("LOCAL_RANK"))  # type: ignore[arg-type]

    model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
    model.fc = nn.Linear(512, 51)
    model = DistributedDataParallel(model.cuda(rank), device_ids=[rank])
    loss = nn.CrossEntropyLoss().cuda(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr)
    scheduler = StepLR(optimizer, 10)

    transform = Compose([
        Resize((112, 112), antialias=True),
        Lambda(lambda x: x/255.0),
        Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225), inplace=True),
    ])
    trainset = HMDB51(root, "", 0, transform=transform, output_format="TCHW")
    testset = HMDB51(root, "", 0, train=False, transform=transform, output_format="TCHW")

    sampler = DistributedSampler(trainset, batch_size=dist.get_world_size())  # type: ignore[var-annotated]
    trainloader = DataLoader(trainset, sampler=sampler)  # type: ignore[arg-type,var-annotated]
    testloader = DataLoader(testset)  # type: ignore[arg-type,var-annotated]

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        model.train()

        tic = time.time()
        for video, _, label in tqdm(trainloader, desc="Epoch {}/{}".format(epoch+1, epochs), ascii=" >="):
            video = video.transpose(1, 2).cuda(rank)
            label = nn.functional.one_hot(label, num_classes=51).float().cuda(rank)
            optimizer.zero_grad()
            loss(model(video), label).backward()
            optimizer.step()
        toc = time.time()
        logging.info("Epoch: {:2d} Elapsed time: {:.4f}".format(epoch, toc - tic))

        if dist.get_rank() == 0:
            model.eval()
            correct = 0
            with torch.no_grad():
                for video, _, label in tqdm(testloader, ascii=" >="):
                    video = video.transpose(1, 2).cuda(rank)
                    label = label.cuda(rank)
                    correct += model(video).argmax(dim=1).eq(label).sum().item()
            logging.info("Epoch: {:2d} Accuracy: {:.4f}".format(epoch, correct / len(testloader.dataset)))  # type: ignore[arg-type]

        scheduler.step()
    dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed training with R(2+1)D and HMDB")
    parser.add_argument("--epochs", default=45, type=int, help="number of epochs to train (default: 45)")
    parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate (default: 0.01)")
    parser.add_argument("--root", default="dataset", type=str, help="root directory of dataset (default: \"dataset\")")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(filename)s:%(lineno)s: %(message)s")
    logging.info(args)
    run(args.epochs, args.lr, args.root)
