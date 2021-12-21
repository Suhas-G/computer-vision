from typing import List, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, show
import torchvision

DEVICE = 'cuda'

def train_one_epoch(dataloader: DataLoader, model: nn.Module, optimizer: SGD,
    writer: SummaryWriter, epoch: int):
    count = 0
    total_loss = 0
    model.train()
    for images, targets in dataloader:
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += float(losses.item())
        count += 1
        
        if count % 500 == 0:
            print("Batch:", count, "Loss:", total_loss)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    total_loss = total_loss / count
    writer.add_scalar('Loss/train', total_loss, global_step=epoch)
    writer.flush()
    return total_loss


def validate(dataloader: DataLoader, model: nn.Module, writer: SummaryWriter, epoch: int):
    # model.eval()
    count = 0
    total_loss = 0
    for images, targets in dataloader:
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        with torch.no_grad():
             loss_dict = model(images, targets)
        
        losses = sum(loss for loss in loss_dict.values())
        total_loss += float(losses.item())
        count += 1

    total_loss = total_loss / count
    writer.add_scalar('Loss/valid', total_loss, global_step=epoch)
    writer.flush()
    return total_loss
        


def train(train_dataloader: DataLoader, val_dataloader: DataLoader, 
            model: nn.Module, optimizer: SGD, lr_scheduler: StepLR, start_epoch, epochs) -> None:
    writer = SummaryWriter()
    model.to(DEVICE)
    for epoch in range(start_epoch, start_epoch + epochs + 1):
        loss = train_one_epoch(train_dataloader, model, optimizer, writer, epoch)
        lr_scheduler.step()
        save_checkpoint(model, optimizer, lr_scheduler, epoch, loss, 'models/checkpoint.pt')
        val_loss = validate(val_dataloader, model, writer, epoch)
    writer.close()
    
    
def draw_boxes(images: List[torch.Tensor], outputs: Dict, score_threshold:float, iou_threshold: float):
    drawn_images = []
    for output, image in zip(outputs, images):
        image = (image * 255).type(torch.uint8)
        confident_indices = (output['scores'] > score_threshold).nonzero().flatten()
        boxes = output['boxes'][confident_indices]
        scores = output['scores'][confident_indices]
        valid_boxes = torchvision.ops.nms(boxes, scores, iou_threshold)
        drawn_images.append(torchvision.utils.draw_bounding_boxes(image.cpu(), boxes[valid_boxes], width=3, colors='red'))
        
    return drawn_images
    
    
def test(model: nn.Module, test_data: DataLoader, score_threshold = 0.8, iou_threshold = 0.5):
    model.to(DEVICE)
    model.eval()
    for test_images in test_data:
        images = test_images[0]
        images = [image.to(DEVICE) for image in images]
        outputs = model(images)
        detected = draw_boxes(images, outputs, score_threshold, iou_threshold)
        show(detected, figsize=(16, 8))
    
    
    
    