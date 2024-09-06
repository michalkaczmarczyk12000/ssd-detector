import torch
import torchvision.transforms as transforms
from torchvision.models.detection import ssd300_vgg16
from torchvision.datasets import CocoDetection
from PIL import Image, ImageDraw
from tqdm import tqdm
import torch.optim as optim
import os


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def train_one_epoch(model, train_loader, optimizer, device, grad_clip=None):
    model.train()
    running_loss = 0.0

    for images, targets in tqdm(train_loader, desc="Trening"):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        loss_dict = model(images, targets)

        if isinstance(loss_dict, dict):
            losses = sum(loss for loss in loss_dict.values())
        elif isinstance(loss_dict, list):
            losses = sum(loss_dict)
        else:
            raise ValueError("Nieoczekiwany typ dla loss_dict")

        if torch.isnan(losses):
            print("Wartość straty to NaN. Pomijam ten krok.")
            continue

        losses.backward()

        if grad_clip:
            clip_gradient(optimizer, grad_clip)

        optimizer.step()

        running_loss += losses.item()

    epoch_loss = running_loss / len(train_loader)
    print(f"Strata po epoce treningowej: {epoch_loss}")
    return epoch_loss


def validate_one_epoch(model, valid_loader, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, targets in tqdm(valid_loader, desc="Walidacja"):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            if isinstance(loss_dict, dict):
                losses = sum(
                    loss
                    for loss in loss_dict.values()
                    if isinstance(loss, torch.Tensor)
                )
            elif isinstance(loss_dict, list):
                losses = sum(loss_dict)
            else:
                raise ValueError("Unexpected type for loss_dict")

            if torch.isnan(losses):
                print("Loss value is NaN. Skipping this step.")
                continue

            running_loss += losses.item()

    epoch_loss = running_loss / len(valid_loader)
    print(f"Validation loss after epoch: {epoch_loss}")
    return epoch_loss


def run_inference(image_path, model, transform):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor.to(device))

    return outputs, image


def display_results(outputs, image):
    draw = ImageDraw.Draw(image)
    for i in range(len(outputs[0]["boxes"])):
        box = outputs[0]["boxes"][i].tolist()
        label = outputs[0]["labels"][i].item()
        score = outputs[0]["scores"][i].item()

        if score > 0.5:
            draw.rectangle(box, outline="red", width=2)
            draw.text(
                (box[0], box[1]), f"Class: {label}, Score: {score:.2f}", fill="red"
            )

    image.show()


def custom_collate_fn(batch):
    images, targets = zip(*batch)
    images = list(images)
    targets = list(targets)
    return images, targets


model = ssd300_vgg16(pretrained=True)
model_path = "model_epoch_42.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))


num_classes = 2
model.head.classification_head.num_classes = num_classes


transform = transforms.Compose(
    [
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ]
)


class PKLotCOCODataset(CocoDetection):
    def __init__(self, root, annFile, transform=None):
        super(PKLotCOCODataset, self).__init__(root, annFile)
        self.transform = transform

    def __getitem__(self, idx):
        while True:
            img, target = super(PKLotCOCODataset, self).__getitem__(idx)

            if self.transform is not None:
                img = self.transform(img)

            boxes = []
            labels = []
            for t in target:
                bbox = t["bbox"]

                if bbox[2] > 0 and bbox[3] > 0:
                    x_min, y_min, width, height = bbox
                    x_max = x_min + width
                    y_max = y_min + height
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(t["category_id"])

            if len(boxes) == 0:
                idx = (idx + 1) % len(self)
                continue

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            target = {"boxes": boxes, "labels": labels}

            return img, target


train_dataset = PKLotCOCODataset(
    root="data/train", annFile="data/train/_annotations.coco.json", transform=transform
)
valid_dataset = PKLotCOCODataset(
    root="data/valid", annFile="data/valid/_annotations.coco.json", transform=transform
)
test_dataset = PKLotCOCODataset(
    root="data/test", annFile="data/test/_annotations.coco.json", transform=transform
)


train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn
)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)


num_epochs = 50
grad_clip = 5.0


for epoch in range(num_epochs):
    print(f"Epoka {epoch + 1}/{num_epochs}")
    train_loss = train_one_epoch(model, train_loader, optimizer, device, grad_clip)
    # valid_loss = validate_one_epoch(model, valid_loader, device)
    print(f"Strata treningowa: {train_loss}")
    torch.save(model.state_dict(), f"model_epoch_{epoch+41}.pth")

print("Trening zakończony.")
