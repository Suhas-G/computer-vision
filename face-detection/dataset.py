from pathlib import Path
from typing import Dict, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor
from torchvision.transforms.transforms import Resize


WIDTH = 1200
HEIGHT = 800


def collate_fn(batch):
    # images, targets = zip(*batch)
    # return (torch.tensor(images), targets)
    return tuple(zip(*batch))

def resize_image(img: Image.Image):
    w, h = img.size
    aspect_ratio = w / h
    if w > h:
        img = img.resize((round(aspect_ratio * 600), 600))
    else:
        img = img.resize((600, round(600 / aspect_ratio)))
    new_aspect_ratio = (img.size[0] / img.size[1])
    assert abs(aspect_ratio - new_aspect_ratio) < 1e-3, f'{aspect_ratio - new_aspect_ratio}, {w, h}, {img.size}'
    return img, (img.size[0] / w), (img.size[1] / h)

class FaceDemoDataset(Dataset):
    def __init__(self, root, transforms = None) -> None:
        super().__init__()
        self.root = root
        self.images = list(map(str, [path.relative_to(Path(root)) for path in Path(root).glob('*.jpg')]))
        self.transforms = transforms
        
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, index) -> torch.Tensor:
        img = Image.open(Path(self.root, self.images[index])).convert("RGB")
        w_ratio, h_ratio = WIDTH / img.size[0] , HEIGHT / img.size[1]
        img = img.resize((WIDTH, HEIGHT))
        if self.transforms is not None:
            img = self.transforms(img)
        return (img, )
        

class FaceDataset(Dataset):
    def __init__(self, root, config_file, transforms = None) -> None:
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.images = []
        with Path(config_file).open() as file:
            while True:
                data = {}
                data['img'] = file.readline().strip()
                if len(data['img']) == 0: break
                data['boxes'] = []
                data['areas'] = []
                no_objects = int(file.readline())
                # If no objects are present, then also theres an extra line in file with zeroes
                if no_objects == 0: 
                    file.readline()
                    continue
                i = 0
                while i < no_objects:
                    x1, y1, w, h, *_ = map(int, file.readline().split())
                    if w > 0 and h > 0 and w * h > 100:
                        data['boxes'].append([x1, y1, w, h])
                    i += 1
                if len(data['boxes']) > 0:
                    self.images.append(data)

    def __len__(self) -> int:
        return len(self.images)


    def __getitem__(self, index) -> Tuple[torch.Tensor, Dict]:
        data = self.images[index]
        img = Image.open(Path(self.root, data['img'])).convert("RGB")
        # img, w_ratio, h_ratio = resize_image(img)
        w_ratio, h_ratio = WIDTH / img.size[0] , HEIGHT / img.size[1]
        img = img.resize((WIDTH, HEIGHT))
        resized_boxes = [[int(x * w_ratio) , int(y * h_ratio), int(w * w_ratio), int(h * h_ratio)] for x,y,w,h in data['boxes']]
        areas = torch.as_tensor([w*h for _,_, w, h in resized_boxes], dtype=torch.float32)
        boxes = torch.as_tensor([[x, y, x + w, y + h] for x, y, w, h in resized_boxes], dtype=torch.int32)
        labels = torch.ones((len(boxes), ), dtype=torch.int64)
        target = {'boxes': boxes, 'areas': areas, 'labels': labels}
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target

def get_dataloader(root: Union[str, Path], config_file: Union[str, Path] = None, transforms: Compose = None, 
    batch_size: int = 4, shuffle: bool = True, demo = False) -> DataLoader:
    if transforms is None:
        transforms = Compose([ToTensor()])
    if demo:
        dataset = FaceDemoDataset(root, transforms)
    else:
        dataset = FaceDataset(root, config_file, transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)