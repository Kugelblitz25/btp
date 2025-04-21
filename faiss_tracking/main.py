import cv2
import numpy as np
import torch
from ultralytics import YOLO
import face_recognition
import faiss
from datetime import datetime
from RealESRGAN import RealESRGAN


class PersonReIDSystem:
    def __init__(self):
        # Load face recognition model
        self.face_model = face_recognition
        
        # Load YOLO model for person detection
        self.yolo_model = YOLO("models/yolov8n-pose.pt")
        
        self.upscalar = RealESRGAN(torch.device('cuda'), scale=4)
        self.upscalar.load_weights('weights/RealESRGAN_x4.pth', download=True)
        
        # Initialize FAISS index for face embeddings (128-dimensional)
        self.face_dim = 128
        self.face_index = faiss.IndexFlatL2(self.face_dim)
        
        # Initialize storage for person IDs and additional data
        self.person_ids = []
        self.last_seen = {}
        self.first_detected = {}
        self.hair_data = {}  # Store hair information by person_id
        
        # Counter for generating new person IDs
        self.next_person_id = 0
    
    def extract_face_features(self, image):
        """Extract face embeddings using face_recognition library"""
        # Convert to RGB (face_recognition requires RGB)
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
            
        # Detect face locations
        face_locations = self.face_model.face_locations(rgb_image)
        
        if not face_locations:
            return None, None
        
        # Get face encodings
        face_encodings = self.face_model.face_encodings(rgb_image, face_locations)
        
        if not face_encodings:
            return None, None
            
        return face_encodings, face_locations
    
    def extract_hair_features(self, image, face_location):
        """Extract hair features based on the top of the face"""
        top, right, bottom, left = face_location
        
        # Estimate hair region (above face)
        hair_top = max(0, top - (bottom - top))
        hair_bottom = top
        hair_left = max(0, left - 10)
        hair_right = min(image.shape[1], right + 10)
        
        if hair_top >= hair_bottom or hair_left >= hair_right:
            return None, None
            
        hair_region = image[hair_top:hair_bottom, hair_left:hair_right]
        
        if hair_region.size == 0:
            return None, None
            
        # Convert to HSV for better color analysis
        hsv_hair = cv2.cvtColor(hair_region, cv2.COLOR_BGR2HSV)
        
        # Calculate average color
        avg_color = np.mean(hsv_hair, axis=(0, 1))
        
        # Simple hair color classification
        hue = avg_color[0]
        saturation = avg_color[1]
        value = avg_color[2]
        
        # Hair color determination
        if value < 50:
            hair_color = "black"
        elif hue < 20 and saturation > 50:
            hair_color = "red"
        elif hue < 20:
            if value < 100:
                hair_color = "brown"
            else:
                hair_color = "blonde"
        else:
            hair_color = "other"
            
        # Hair length estimation based on region size
        hair_length = "short" if (hair_bottom - hair_top) < (bottom - top) * 0.3 else "long"
        
        return hair_color, hair_length
    
    def store_features(self, person_id, face_embedding, hair_data):
        """Store extracted features in FAISS index"""
        # Store face embedding in FAISS
        if face_embedding is not None:
            # Convert to float32 for FAISS
            face_embedding_f32 = face_embedding.astype(np.float32)
            
            # Add to FAISS index
            self.face_index.add(np.array([face_embedding_f32]))
            self.person_ids.append(person_id)
        
        # Store hair data
        if hair_data is not None:
            hair_color, hair_length = hair_data
            if person_id not in self.hair_data:
                self.hair_data[person_id] = []
            
            self.hair_data[person_id].append({
                "color": hair_color,
                "length": hair_length,
                "timestamp": datetime.now()
            })
        
        # Update timestamps
        self.last_seen[person_id] = datetime.now()
        if person_id not in self.first_detected:
            self.first_detected[person_id] = datetime.now()
    
    def identify_person(self, face_embedding, hair_data, threshold=0.6):
        """Match a person against FAISS index"""
        if face_embedding is None or len(self.person_ids) == 0:
            return None, 0.0
        
        # Convert to float32 for FAISS
        face_embedding_f32 = face_embedding.astype(np.float32)
        
        # Search in FAISS
        distances, indices = self.face_index.search(np.array([face_embedding_f32]), 1)
        
        if indices[0][0] >= len(self.person_ids):
            return None, 0.0
            
        # Get the closest match
        distance = distances[0][0]
        person_id = self.person_ids[indices[0][0]]
        
        # Convert distance to similarity (lower distance = higher similarity)
        face_similarity = 1.0 / (1.0 + distance)
        
        # If face similarity is high enough, return the match
        if face_similarity >= threshold:
            return person_id, face_similarity
            
        # If not confident enough with face alone, check hair data
        if hair_data is not None and person_id in self.hair_data:
            hair_color, hair_length = hair_data
            
            # Check against stored hair data
            hair_matches = []
            for stored_hair in self.hair_data[person_id]:
                db_hair_color = stored_hair["color"]
                db_hair_length = stored_hair["length"]
                
                # Calculate hair similarity
                color_match = 1.0 if hair_color == db_hair_color else 0.0
                length_match = 1.0 if hair_length == db_hair_length else 0.0
                
                # Calculate overall hair match
                hair_match = color_match * 0.6 + length_match * 0.4
                hair_matches.append(hair_match)
            
            # Get best hair match
            hair_similarity = max(hair_matches) if hair_matches else 0.0
            
            # Combine face and hair similarity
            combined_similarity = face_similarity * 0.8 + hair_similarity * 0.2
            
            if combined_similarity >= threshold:
                return person_id, combined_similarity
        
        return None, face_similarity
    
    def register_new_person(self, face_embedding, hair_data):
        """Register a new person in the system"""
        person_id = self.next_person_id
        self.next_person_id += 1
        
        self.store_features(person_id, face_embedding, hair_data)
        return person_id
    
    def process_frame(self, frame):
        results = []
        
        # Use YOLO to detect people
        yolo_results = self.yolo_model(frame)

        inside = lambda x, y: (0 <= x <= frame.shape[0]) & (0 <= y <= frame.shape[1])
        
        # Extract person detections
        person_boxes = []
        for result in yolo_results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]  # get box coordinates
                confidence = box.conf[0]  # get confidence
                class_id = box.cls[0]  # get class id
                if class_id == 0 and confidence > 0.9 and inside(x1, y1) and inside(x2, y2):
                    person_boxes.append([int(x1), int(y1), int(x2), int(y2)])
        
        # Extract face features for each detected person
        for bbox in person_boxes:
            x1, y1, x2, y2 = bbox
            person_img = frame[y1:y2, x1:x2]
            
            if person_img.size == 0:
                continue

            person_img = np.asarray(self.upscalar.predict(person_img))
            print(person_img.shape)

            # denoised = cv2.fastNlMeansDenoisingColored(person_img, 
            #     None, 5, 5, 7, 7
            # )
            
            # Enhance contrast using CLAHE
            if len(person_img.shape) == 3 and person_img.shape[2] == 3:
                lab = cv2.cvtColor(person_img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
                l = clahe.apply(l)
                lab = cv2.merge((l, a, b))
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
                enhanced = clahe.apply(person_img)
            
            # Apply sharpening
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            person_img = cv2.filter2D(enhanced, -1, kernel)

            cv2.imwrite(f"faiss_tracking/images/{str(datetime.now())}.jpg", person_img)
                
            # # Extract face features
            face_embeddings, face_locations = self.extract_face_features(person_img)
            
            # If no face detected, skip this detection
            if face_embeddings is None:
                continue
                
            face_embedding = face_embeddings[0]  # Use first face if multiple detected
            face_location = face_locations[0]
            
            # Extract hair features
            hair_data = self.extract_hair_features(person_img, face_location)
            
            # Try to identify person
            person_id, confidence = self.identify_person(face_embedding, hair_data)
            
            if person_id is None:
                # Register new person
                person_id = self.register_new_person(face_embedding, hair_data)
                is_new = True
            else:
                # Update existing person features
                self.store_features(person_id, face_embedding, hair_data)
                is_new = False
            
            # Adjust face location to global coordinates
            global_face_location = (
                face_location[0] + y1,
                face_location[1] + x1,
                face_location[2] + y1,
                face_location[3] + x1
            )
            
            results.append({
                "person_id": 1,
                "bbox": bbox,
                "face_location": global_face_location,
                "confidence": confidence,
                "is_new": is_new
            })

        print(len(results) - len(person_boxes))
        
        return results


# Example usage
if __name__ == "__main__":
    # Initialize the system
    reid_system = PersonReIDSystem()
    
    # Example with webcam or video file
    cap = cv2.VideoCapture("test_videos/test_vid_from_ncair_1.mp4")  # Use 0 for webcam or file path for video
    
    # Get video properties for output
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('reid_output.avi', fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame to detect and identify people
        results = reid_system.process_frame(frame)
        
        # Visualize results
        for result in results:
            x1, y1, x2, y2 = result["bbox"]
            person_id = result["person_id"]
            confidence = result["confidence"]
            
            # Draw rectangle around person
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw rectangle around face
            if "face_location" in result:
                top, right, bottom, left = result["face_location"]
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            
            # Display person ID and confidence
            cv2.putText(frame, f"ID: {person_id} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Write the frame to the output file
        out.write(frame)
    
    # Release everything when done
    cap.release()
    out.release()