import insightface
from insightface.app import FaceAnalysis

def load_insightface():
    app = FaceAnalysis(providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(224, 224))
    return app

def extract_embedding(image, app):
    faces = app.get(image)
    if len(faces) > 0:
        return faces[0].embedding
    return [-1]