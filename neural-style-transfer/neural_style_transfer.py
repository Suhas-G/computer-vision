from typing import List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from config import (
    CONTENT_CONTENT_LAYER,
    CONTENT_IMAGE,
    CONTENT_LEARNING_RATE,
    CONTENT_LR_SCHEDULER_DECAY,
    CONTENT_LR_SCHEDULER_STEP,
    CONTENT_WEIGHT,
    EPOCHS,
    HEIGHT,
    IMAGE_VISUALISE_FREQUENCY,
    STYLE_IMAGE,
    STYLE_LEARNING_RATE,
    STYLE_LR_SCHEDULER_DECAY,
    STYLE_LR_SCHEDULER_STEP,
    STYLE_STYLE_LAYERS,
    STYLE_WEIGHT,
    TRANSFER_CONTENT_LAYER,
    TRANSFER_LEARNING_RATE,
    TRANSFER_LR_SCHEDULER_DECAY,
    TRANSFER_LR_SCHEDULER_STEP,
    TRANSFER_STARTING_IMAGE_TYPE,
    TRANSFER_STYLE_LAYERS,
    WIDTH,
    StartingImage,
)
from models import VGG16

writer = SummaryWriter()

IMAGENET_MEAN_255 = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32)
IMAGENET_STD_NEUTRAL = torch.tensor([1, 1, 1], dtype=torch.float32)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_image_from_tensor(image: torch.Tensor) -> torch.Tensor:
    output = np.copy(image.cpu().detach())
    output = output + np.array(IMAGENET_MEAN_255).reshape((3, 1, 1))
    output = torch.clip(torch.from_numpy(output), 0, 255)
    return output.type(torch.uint8)


def load_image(path) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((WIDTH, HEIGHT))
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255)),
            transforms.Normalize(
                mean=IMAGENET_MEAN_255.tolist(), std=IMAGENET_STD_NEUTRAL.tolist()
            ),
        ]
    )

    img = transform(img).unsqueeze(0)
    return img


def get_variable_image(image_tensor: torch.Tensor) -> Variable:
    return Variable(image_tensor.type(torch.float32).to(DEVICE), requires_grad=True)


def get_content_loss(
    image_features: torch.Tensor, target_features: torch.Tensor
) -> torch.Tensor:
    mse_loss = nn.MSELoss()
    return mse_loss(image_features, target_features)


def reconstruct_content(image: torch.Tensor, target: torch.Tensor) -> None:

    global writer

    optimizer = torch.optim.Adam((image,), lr=CONTENT_LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, CONTENT_LR_SCHEDULER_STEP, gamma=CONTENT_LR_SCHEDULER_DECAY
    )
    model = VGG16(content_layer=CONTENT_CONTENT_LAYER).to(DEVICE)

    writer.add_image("Content/Target", get_image_from_tensor(target.squeeze()))
    content_features, _ = model(target)
    content_features = content_features.squeeze()

    for epoch in range(EPOCHS + 1):
        noise_features, _ = model(image)
        loss = get_content_loss(noise_features.squeeze(), content_features)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        writer.add_scalar("Content/Loss", loss.item(), global_step=epoch)

        with torch.no_grad():
            if epoch % IMAGE_VISUALISE_FREQUENCY == 0:
                writer.add_image(
                    "Content/Generated",
                    get_image_from_tensor(image.squeeze()),
                    global_step=epoch,
                )

    writer.add_hparams(
        {
            "Content/LR": CONTENT_LEARNING_RATE,
            "Content/LR_STEP": CONTENT_LR_SCHEDULER_STEP,
            "Content/LR_decay": CONTENT_LR_SCHEDULER_DECAY,
            "Content/Layer": CONTENT_CONTENT_LAYER,
        },
        {"hparams/Content/Loss": loss.item()},
    )


def construct_gram_matrix(feature_map: torch.Tensor) -> torch.Tensor:
    gram_matrix = feature_map @ feature_map.T
    return gram_matrix


def get_style_loss(
    image_features: torch.Tensor, target_gram_matrices: List[torch.Tensor]
) -> torch.Tensor:
    loss = 0.0
    for i, feature in enumerate(image_features):
        feature = feature.squeeze()
        noise_feature_map = feature.view(feature.shape[0], -1)
        noise_gram = construct_gram_matrix(noise_feature_map)

        layer_loss = torch.sum(torch.square((target_gram_matrices[i] - noise_gram))) / (
            4 * (feature.shape[0] ** 2) * (feature.shape[1] ** 2)
        )
        loss += layer_loss / len(image_features)

    return loss


def reconstruct_style(image: torch.Tensor, target: torch.Tensor) -> None:
    global writer
    optimizer = torch.optim.Adam((image,), lr=STYLE_LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, STYLE_LR_SCHEDULER_STEP, gamma=STYLE_LR_SCHEDULER_DECAY
    )
    model = VGG16(style_layers=STYLE_STYLE_LAYERS).to(DEVICE)
    writer.add_image("Style/Target", get_image_from_tensor(target.squeeze()))
    _, style_features = model(target)

    style_grams = []
    for feature in style_features:
        feature = feature.squeeze()
        style_feature_map = feature.view(feature.shape[0], -1)
        style_grams.append(construct_gram_matrix(style_feature_map))

    for epoch in range(EPOCHS + 1):

        _, noise_features = model(image)
        loss = get_style_loss(noise_features, style_grams)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        writer.add_scalar("Style/Loss", loss.item(), global_step=epoch)

        with torch.no_grad():
            if epoch % IMAGE_VISUALISE_FREQUENCY == 0:
                writer.add_image(
                    "Style/Generated",
                    get_image_from_tensor(image.squeeze()),
                    global_step=epoch,
                )

    writer.add_hparams(
        {
            "Style/LR": STYLE_LEARNING_RATE,
            "Style/LR_STEP": STYLE_LR_SCHEDULER_STEP,
            "Style/LR_decay": STYLE_LR_SCHEDULER_DECAY,
            "Style/Layer": " ".join(STYLE_STYLE_LAYERS),
        },
        {"hparams/Style/Loss": loss.item()},
    )


def style_transfer(
    content_image: torch.Tensor, style_image: torch.Tensor, start_type: StartingImage
) -> None:
    global writer
    image = None
    if start_type == StartingImage.CONTENT:
        image = get_variable_image(torch.clone(content_image))
    elif start_type == StartingImage.STYLE:
        image = get_variable_image(torch.clone(style_image))
    else:
        image = get_variable_image(
            torch.randn((1, 3, HEIGHT, WIDTH), dtype=torch.float32)
        )

    optimizer = torch.optim.Adam((image,), lr=TRANSFER_LEARNING_RATE)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, TRANSFER_LR_SCHEDULER_STEP, gamma=TRANSFER_LR_SCHEDULER_DECAY
    )

    model = VGG16(
        content_layer=TRANSFER_CONTENT_LAYER, style_layers=TRANSFER_STYLE_LAYERS
    ).to(DEVICE)

    target_content_features, _ = model(content_image)
    target_content_features = target_content_features.squeeze()

    _, target_style_features = model(style_image)
    target_style_grams = []
    for feature in target_style_features:
        feature = feature.squeeze()
        target_style_feature_map = feature.view(feature.shape[0], -1)
        target_style_grams.append(construct_gram_matrix(target_style_feature_map))

    for epoch in range(EPOCHS + 1):

        content_features, style_features = model(image)
        content_loss = get_content_loss(
            content_features.squeeze(), target_content_features
        )
        style_loss = get_style_loss(style_features, target_style_grams)

        total_loss = (CONTENT_WEIGHT * content_loss) + (STYLE_WEIGHT * style_loss)

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()
        writer.add_scalar(
            "Style Transfer/Content Loss", content_loss.item(), global_step=epoch
        )
        writer.add_scalar(
            "Style Transfer/Style Loss", style_loss.item(), global_step=epoch
        )
        writer.add_scalar("Style Transfer/Loss", total_loss.item(), global_step=epoch)

        with torch.no_grad():
            if epoch % IMAGE_VISUALISE_FREQUENCY == 0:
                writer.add_image(
                    "Style Transfer/Generated",
                    get_image_from_tensor(image.squeeze()),
                    global_step=epoch,
                )

    writer.add_hparams(
        {
            "Transfer/LR": TRANSFER_LEARNING_RATE,
            "Transfer/LR_STEP": TRANSFER_LR_SCHEDULER_STEP,
            "Transfer/LR_decay": TRANSFER_LR_SCHEDULER_DECAY,
            "Transfer/Style Layers": " ".join(TRANSFER_STYLE_LAYERS),
            "Transfer/Content Layer": TRANSFER_CONTENT_LAYER,
            "Transfer/Starting Image": start_type.name,
            "Transfer/Content Weight": CONTENT_WEIGHT,
            "Transfer/Style Weight": STYLE_WEIGHT
        },
        {"hparams/Transfer/Loss": total_loss.item()},
    )


def main(content=True, style=False, transfer=False) -> None:

    noise_image = torch.randn((1, 3, HEIGHT, WIDTH), dtype=torch.float32)
    noise_image = get_variable_image(noise_image)

    if content:
        content_img = load_image(CONTENT_IMAGE).to(DEVICE)
        reconstruct_content(noise_image, content_img)
    elif style:
        style_img = load_image(STYLE_IMAGE).to(DEVICE)
        reconstruct_style(noise_image, style_img)
    elif transfer:
        content_img = load_image(CONTENT_IMAGE).to(DEVICE)
        style_img = load_image(STYLE_IMAGE).to(DEVICE)
        style_transfer(content_img, style_img, TRANSFER_STARTING_IMAGE_TYPE)
    writer.close()


if __name__ == "__main__":
    main(content=True, style=False, transfer=False)
