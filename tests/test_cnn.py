import argparse
import logging
import os
import time
from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import Compose, Lambda, Normalize, Resize

from tests.datasets import HMDB51


def collate(batch: List[Tuple[Tensor, Tensor, int]]) -> Tuple[Tensor, Tensor]:
    videos = list()
    labels = list()
    for video, _, label in batch:
        videos.append(video)
        for _ in range(len(video)):
            labels.append(label)
    return torch.vstack(videos), torch.tensor(labels)

def run(epochs: int, batch_size: int, lr: float):
    dist.init_process_group(backend=dist.Backend.NCCL)

    # environment variables set by torchrun
    local_rank = int(os.getenv("LOCAL_RANK"))  # type: ignore[arg-type]
    rank = int(os.getenv("RANK"))  # type: ignore[arg-type]
    world_size = int(os.getenv("WORLD_SIZE"))  # type: ignore[arg-type]

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(512, 51)
    model.cuda(local_rank)
    model = DistributedDataParallel(model, device_ids=[local_rank])
    loss = nn.CrossEntropyLoss().cuda(local_rank)
    optimizer = torch.optim.SGD(model.parameters(), lr)
    scheduler = StepLR(optimizer, 10)

    transform = Compose([
        Resize((224, 224), antialias=True),
        Lambda(lambda x: x/255.0),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True),
    ])
    trainset = HMDB51("dataset", "", 0, transform=transform, output_format="TCHW")
    testset = HMDB51("dataset", "", 0, train=False, transform=transform, output_format="TCHW")

    sampler = DistributedSampler(trainset)  # type: ignore[var-annotated]
    trainloader = DataLoader(trainset, batch_size=batch_size // world_size, sampler=sampler, collate_fn=collate)
    testloader = DataLoader(testset, batch_size=batch_size // world_size, collate_fn=collate)

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        model.train()

        tic = time.time()
        for frames, labels in trainloader:
            frames = frames.cuda(local_rank)
            labels = nn.functional.one_hot(labels, num_classes=51).cuda(local_rank)
            optimizer.zero_grad()
            loss(model(frames), labels).backward()
            optimizer.step()
        toc = time.time()
        logging.info("Epoch: {:2d} Elapsed time: {:.4f}".format(epoch, toc - tic))

        if rank == 0:
            model.eval()
            correct = 0
            with torch.no_grad():
                for frames, labels in testloader:
                    frames = frames.cuda(local_rank)
                    labels = labels.cuda(local_rank)
                    correct += model(frames).argmax(dim=1).eq(labels).sum().item()
            logging.info("Epoch: {:2d} Accuracy: {:.4f}".format(epoch, correct / len(testloader.dataset)))  # type: ignore[arg-type]

        scheduler.step()
        dist.barrier()
    dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed training with ResNet and HMDB51")
    parser.add_argument("--epochs", default=45, type=int, help="number of epochs to train (default: 45)")
    parser.add_argument("--batch-size", default=32, type=int, help="mini-batch size (default: 32)")
    parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate (default: 0.01)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(filename)s:%(lineno)s: %(message)s")
    logging.info(args)
    run(args.epochs, args.batch_size, args.lr)
