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
            nn.Linear(64, num_classes) 
        )
    def forward(self, x):
        return self.model(x)
    
num_classes = 40
model = ModifiedViT(num_classes=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("C:/Users/vedan/OneDrive - Indian Institute of Technology Bombay/BTP/VIT_weights.pth", map_location=device, weights_only=True))
model.eval()

actions = [
    "applauding", "blowing_bubbles", "brushing_teeth", "cleaning_the_floor", "climbing",
    "cooking", "cutting_trees", "cutting_vegetables", "drinking", "feeding_a_horse",
    "fishing", "fixing_a_bike", "fixing_a_car", "gardening", "holding_an_umbrella",
    "jumping", "looking_through_a_microscope", "looking_through_a_telescope", "playing_guitar",
    "playing_violin", "pouring_liquid", "pushing_a_cart", "reading", "phoning", "riding_a_bike",
    "riding_a_horse", "rowing_a_boat", "running", "shooting_an_arrow", "smoking", "taking_photos",
    "texting_message", "throwing_frisby", "using_a_computer", "walking_the_dog", "washing_dishes",
    "watching_TV", "waving_hands", "writing_on_a_board", "writing_on_a_book"
]

def classify_action(image):
    # To test on an image
    # image = Image.open(image).convert("RGB")
    test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    input_tensor = test_transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted_class = torch.max(outputs, 1)

        print(f"Predicted class index: {actions[predicted_class.item()]}")
# To test on an image
# classify_action("C:/Users/vedan/OneDrive/Documents/btp/mediapipe/Test-Human-1.png")