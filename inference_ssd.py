import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from torchvision.models.detection import ssd300_vgg16


model_path = "full_model_epoch_75.pth"

image_path = "proba3.jpg"


model = ssd300_vgg16(pretrained=False)
num_classes = 2
model.head.classification_head.num_classes = num_classes


try:
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    print("Model załadowany poprawnie.")
except Exception as e:
    print(f"Błąd podczas ładowania modelu: {e}")
    exit(1)

model.eval()


transform = transforms.Compose([transforms.ToTensor()])


def run_inference(image_path, model, transform):
    try:

        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image_tensor)

        return outputs, image
    except FileNotFoundError:
        print(f"Plik {image_path} nie został znaleziony.")
        exit(1)
    except Exception as e:
        print(f"Błąd podczas inferencji: {e}")
        exit(1)


def display_results(outputs, image):
    draw = ImageDraw.Draw(image)
    for i in range(len(outputs[0]["boxes"])):
        box = outputs[0]["boxes"][i].tolist()
        label = outputs[0]["labels"][i].item()
        score = outputs[0]["scores"][i].item()

        if score > 0.01:
            if label == 1:
                draw.rectangle(box, outline="green", width=2)
                label_text = "free"
                draw.text((box[0], box[1]), f"{label_text}", fill="green")
            elif label == 2:
                draw.rectangle(box, outline="red", width=2)
                label_text = "taken"
                draw.text((box[0], box[1]), f"{label_text}", fill="red")
            print(f"Class: {label}, {label_text}")

    image.show()


outputs, image = run_inference(image_path, model, transform)

display_results(outputs, image)
