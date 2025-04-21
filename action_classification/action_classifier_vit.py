import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn


class ModifiedViT(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedViT, self).__init__()

        self.model = models.vit_b_16(weights="IMAGENET1K_V1")

        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.model(x)


def load_vit():
    num_classes = 40
    model = ModifiedViT(num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(
        torch.load(
            "action_classification/VIT_weights.pth",
            map_location=device,
            weights_only=True,
        )
    )
    return model


def classify_action(image, model):
    # To test on an image
    # image = Image.open(image).convert("RGB")

    image = Image.fromarray(image)
    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = test_transform(image).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted_class = torch.max(outputs, 1)

        return predicted_class.item()


# To test on an image
# classify_action("C:/Users/vedan/OneDrive/Documents/btp/mediapipe/Test-Human-1.png")
