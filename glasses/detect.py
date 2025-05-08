from glasses_detector import GlassesClassifier
from PIL import Image

def load_classifier():
    classifier = GlassesClassifier()
    return classifier

def check_glasses(face_crop, classifier):
    face_crop = Image.fromarray(face_crop)
    return classifier.predict(face_crop)