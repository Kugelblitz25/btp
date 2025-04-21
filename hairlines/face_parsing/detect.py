import torch
import numpy as np
import cv2
from hairline.face_parsing.model import BiSeNet
import matplotlib.pyplot as plt

def load_model(model_path='hairline/face_parsing/79999_iter.pth'):
    model = BiSeNet(n_classes=19)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def segment(image, model):
    detected = True
    h, w = image.shape[:2]
    image = cv2.resize(image, (512, 512))
    image = image.astype(np.float32) / 255.0
    image = torch.tensor(image.transpose(2, 0, 1)).unsqueeze(0)

    with torch.no_grad():
        output = model(image)[0]
        parsing = output.squeeze(0).argmax(0).cpu().numpy()

    hair_mask = (parsing == 17).astype(np.uint8) 
    face_mask = (parsing == 1).astype(np.uint8)
    eyebrow_mask = np.logical_or(parsing == 2, parsing == 3).astype(np.uint8)
    if np.sum(eyebrow_mask) > 0:
        eyebrow_coords = np.where(eyebrow_mask > 0)
        eyebrow_top = np.min(eyebrow_coords[0]) 
    else:
        face_coords = np.where(face_mask > 0)
        if len(face_coords[0]) > 0:
            face_top = np.min(face_coords[0])
            face_bottom = np.max(face_coords[0])
            eyebrow_top = face_top + (face_bottom - face_top) // 3
        else:
            detected = False
            return None, None, detected

    head_classes = [1, 4, 5, 10, 13, 14]
    full_head_mask = np.isin(parsing, head_classes).astype(np.uint8)
    head_mask = np.zeros_like(full_head_mask)
    head_mask[:eyebrow_top, :] = full_head_mask[:eyebrow_top, :] 
    kernel = np.ones((5, 5), np.uint8)
    head_mask = cv2.morphologyEx(head_mask, cv2.MORPH_CLOSE, kernel)

    hair_mask = cv2.resize(hair_mask, (w, h)) * 255
    head_mask = cv2.resize(head_mask, (w, h)) * 255 + hair_mask
    
    return hair_mask, head_mask, detected

def overlay_mask(image, hair_mask,head_mask):
    hair_overlay = np.zeros_like(image)
    head_overlay = np.zeros_like(image)
    hair_overlay[:, :, 2] = hair_mask
    head_overlay[:, :, 1] = head_mask 
    result = cv2.addWeighted(image, 1, hair_overlay, 0.4, 0)
    result = cv2.addWeighted(result, 0.8, head_overlay, 0.2, 0)
    return result

def compute_baldness(hair_mask, head_mask):
    total_head_pixels = np.sum(head_mask > 0)  
    total_hair_pixels = np.sum(hair_mask > 0)  
    if total_head_pixels == 0:
        return 0 
    hair_percentage = (total_hair_pixels / total_head_pixels) * 100
    return hair_percentage

def load_bisenet():
    model = load_model("hairline/face_parsing/79999_iter.pth") 
    return model

def Hairline(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hair_mask, head_mask, detected = segment(image, model)
    if not detected:
        # print("Face not detected clearly")
        hair_percentage = -1
    else:
        hair_percentage = compute_baldness(hair_mask, head_mask)
    return hair_percentage



# To check the segmented hairline uncomment the code below

# image_path = "/home/hkavediya/BTP/btp/images/image_4.png" 
# image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# model = load_model("hairline/face_parsing/79999_iter.pth")  
# hair_mask, head_mask = segment(image, model)
# hair_percentage = compute_baldness(hair_mask, head_mask)
# print(f"Estimated hair: {hair_percentage:.2f}%")
# result = overlay_mask(image, hair_mask, head_mask)
# plt.figure(figsize=(10, 5))
# plt.imshow(result)
# plt.axis("off")
# plt.imsave("output.png", result)
# print("Image saved as output.png")



