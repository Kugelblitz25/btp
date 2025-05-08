from models.camera import Camera
from models.parsing import HumanProcessor
from pipeline.models.track import PersonTracker
from hairlines.face_parsing.detect import Hairline, load_bisenet
import json
import cv2
from models.database import *
from action_classification.action_classifier_vit import classify_action, load_vit
from insight_face.reidentification import extract_embedding, load_insightface
from fairface.predict import load_fairface, predict_age_gender_race
from glasses.detect import load_classifier, check_glasses
import base64
import io
from PIL import Image
from collections import defaultdict
import requests
from datetime import datetime

with open("pipeline/tracker_config.json") as f:
    config = json.load(f)

def base_64(face_crop):
    buffer = io.BytesIO()
    face_crop = Image.fromarray(face_crop)
    face_crop.save(buffer, format="JPEG")  
    buffer.seek(0)  
    face_crop_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return face_crop_base64

def get_hairline_type(hair_percentage):
    if hair_percentage >= 85:
        return 1  # Full Hair Coverage
    elif hair_percentage >= 65:
        return 2  # Early Thinning
    elif hair_percentage >= 45:
        return 3  # Diffuse Thinning
    elif hair_percentage >= 25:
        return 4  # Partial Baldness
    elif hair_percentage >= 5:
        return 5  # Extensive Baldness
    else:
        return 6  # Near-Total Baldness



tracker = PersonTracker(config)
parser = HumanProcessor()
# vid_loc = "rtsp://ncair:ncair@10.185.153.173/cam/realmonitor?channel=2&subtype=1"
vid_loc = "test_videos/test_vid_from_ncair_1.mp4"

conf_thres = 0.8 # Confidence threshold of being human
race_thres = 0.3
age_thres = 0.4
gender_thres = 0.8
pos_thresh = 10.0
count = 0

actions = [
    "applauding",
    "blowing_bubbles",
    "brushing_teeth",
    "cleaning_the_floor",
    "climbing",
    "cooking",
    "cutting_trees",
    "cutting_vegetables",
    "drinking",
    "feeding_a_horse",
    "fishing",
    "fixing_a_bike",
    "fixing_a_car",
    "gardening",
    "holding_an_umbrella",
    "jumping",
    "looking_through_a_microscope",
    "looking_through_a_telescope",
    "playing_guitar",
    "playing_violin",
    "pouring_liquid",
    "pushing_a_cart",
    "reading",
    "phoning",
    "riding_a_bike",
    "riding_a_horse",
    "rowing_a_boat",
    "running",
    "shooting_an_arrow",
    "smoking",
    "taking_photos",
    "texting_message",
    "throwing_frisby",
    "using_a_computer",
    "walking_the_dog",
    "washing_dishes",
    "watching_TV",
    "waving_hands",
    "writing_on_a_board",
    "writing_on_a_book",
]

idx_race = {
    1: "White",
    2: "Black",
    3: "Latino_Hispanic",
    4: "East Asian",
    5: "Southeast Asian",
    6: "Indian",
    7: "Middle Eastern",
}

idx_age = {
    1: "0-2",
    2: "3-9",
    3: "10-19",
    4: "20-29",
    5: "30-39",
    6: "40-49",
    7: "50-59",
    8: "60-69",
    9: "70+",
}

idx_gender = {1: "Male", 2: "Female"}
id_db = defaultdict(int)

vit = load_vit()
bisenet = load_bisenet()
fair_face = load_fairface()
app = load_insightface()
classifier = load_classifier()

with Camera(vid_loc) as camera:
    for frame in camera.frames():

        results = tracker.track(frame)
        frame, results = parser.process_frame(frame, results)

        for id in results:
            
            confidence = results[id]["confidence"] # Confidence of being human
            
            if confidence >= conf_thres: # Human detected
                hairline_type = race = age = gender = features = base64_features = None
                glasses = False 

                bbox = results[id]["bbox"]
                left, top, right, bottom = map(int, bbox)
                face_bbox = results[id]["face_bbox"]
                f_left, f_top, f_right, f_bottom = map(int, face_bbox)
                face_crop = frame[f_top:f_bottom, f_left:f_right]
                
                x = (results[id]["foot_coordinates"]["left"][0]+ results[id]["foot_coordinates"]["right"][0]) / 2.0
                y = (results[id]["foot_coordinates"]["left"][1]+ results[id]["foot_coordinates"]["right"][1]) / 2.0
                
                pant = results[id]["lower_body_color"]["name"]
                shirt = results[id]["torso_color"]["name"]
                height = int(1 * (bottom-top))

                hair_percentage = Hairline(face_crop, bisenet)         

                if hair_percentage != -1 : # Face detected clearly

                    hairline_type = get_hairline_type(hair_percentage)
                    race, age, gender, race_score, age_score, gender_score = (predict_age_gender_race(face_crop, fair_face))
                    
                    if race_score >= race_thres and  gender_score>=gender_thres and age_score>=age_thres:
                        features = extract_embedding(face_crop, app)

                        if len(features)!= 1: # Features extracted successfully
                            base64_features = base_64(face_crop)
                            action_id = classify_action(frame[top:bottom, left:right], vit)
                            glasses = True if check_glasses(face_crop, classifier) == "present" else False                
                        else:
                            features = None
                    else:
                        age = gender = race = None  

                person_data = {
                "base64": base64_features,
                "height": height,
                "glasses": glasses,
                "feature": json.dumps(features.tolist())  if features is not None else "",
                "gender_id": gender ,
                "hairline_id": hairline_type ,
                "race_id": race ,
                "age_id": age 
                }


                if id not in id_db: # New person
                    person_response = requests.post("https://dbapi-2zb1.onrender.com/persons", json=person_data)
                    if not person_response.status_code == 200:
                        print(person_response.json())
                    if person_response.status_code == 200:
                        person_key = person_response.json()["id"]
                        id_db[id] = person_key

                        track_data = {
                        "person_id": person_key,
                        "time": datetime.utcnow().isoformat(),
                        "duration": "PT0.1S",  
                        "x": x,
                        "y": y
                        }
                        track_resp = requests.post("https://dbapi-2zb1.onrender.com/tracks/", json=track_data)
                        print("Posted new person", id, track_resp.json())
                        apparel_data = {
                            "person_id": person_key,
                            "shirt_colour": shirt,
                            "pant_colour": pant,
                            "time": datetime.utcnow().isoformat()
                        }
                        
                        resp = requests.post("https://dbapi-2zb1.onrender.com/apparels/", json=apparel_data)
                        print("Apparel new person", resp.json())
                        
                        if 'action_id' in locals() and action_id is not None:
                            event_data = {
                                "person_id": person_key,
                                "action_id": action_id + 1,  
                                "time": datetime.utcnow().isoformat()
                            }
                            requests.post("https://dbapi-2zb1.onrender.com/events/", json=event_data)

                else:
                    person_key = id_db[id]
                    person_response = requests.get(f"https://dbapi-2zb1.onrender.com/persons/{person_key}").json()
                    person_data = {key: person_data[key] or person_response[key] for key in person_data}
                    resp = requests.patch(f"https://dbapi-2zb1.onrender.com/persons/{person_key}", json=person_data)
                    print("Update Person",id,resp.json())
                    tracks_response = requests.get(f"https://dbapi-2zb1.onrender.com/tracks/?person_id={person_key}")

                    
                    if tracks_response.status_code == 200:
                        tracks = tracks_response.json()
                        current_time = datetime.utcnow()
                        update_track = False

                        if tracks:
                            latest_track = tracks[-1]
                            track_time = datetime.fromisoformat(latest_track["time"])
                            time_diff = (current_time - track_time).total_seconds()
                            
                            if (abs(latest_track["x"] - x) <= pos_thresh and 
                                abs(latest_track["y"] - y) <= pos_thresh):

                                update_track = True
                                duration = f"PT{time_diff}S"
                                track_data = {
                                    "person_id": person_key,
                                    "time": track_time.isoformat(),
                                    "duration": duration,
                                    "x": x,
                                    "y": y
                                }
                                resp = requests.patch(f"https://dbapi-2zb1.onrender.com/tracks/{latest_track['id']}", json=track_data)
                                print("Update track",id,resp.json())
                        
                        if not update_track:
                            track_data = {
                                "person_id": person_key,
                                "time": current_time.isoformat(),
                                "duration": "PT0.1S",
                                "x": x,
                                "y": y
                            }
                            resp = requests.post("https://dbapi-2zb1.onrender.com/tracks/", json=track_data)
                            print("Update rack",id,resp.json())
                    
                    if 'action_id' in locals() and action_id is not None:
                        event_data = {
                            "person_id": person_key,
                            "action_id": action_id + 1,
                            "time": datetime.utcnow().isoformat()
                        }
                        requests.post("https://dbapi-2zb1.onrender.com/events/", json=event_data)
        
            else:
                continue