from models.camera import Camera
from models.parsing import HumanProcessor
from pipeline.models.track import PersonTracker
from hairlines.face_parsing.detect import Hairline, load_bisenet
import json
import cv2
from models.database import *
from action_classification.action_classifier_vit import classify_action, load_vit
from fairface.predict import load_fairface, predict_age_gender_race

with open("pipeline/tracker_config.json") as f:
    config = json.load(f)


tracker = PersonTracker(config)
parser = HumanProcessor()
vid_loc = "rtsp://ncair:ncair@10.185.153.173/cam/realmonitor?channel=2&subtype=1"
# vid_loc = "test_videos/test_vid_from_ncair_1.mp4"
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
    0: "White",
    1: "Black",
    2: "Latino_Hispanic",
    3: "East Asian",
    4: "Southeast Asian",
    5: "Indian",
    6: "Middle Eastern",
}

idx_age = {
    0: "0-2",
    1: "3-9",
    2: "10-19",
    3: "20-29",
    4: "30-39",
    5: "40-49",
    6: "50-59",
    7: "60-69",
    8: "70+",
}

idx_gender = {0: "Male", 1: "Female"}

vit = load_vit()
bisenet = load_bisenet()
fair_face = load_fairface()

with Camera(vid_loc) as camera:
    for frame in camera.frames():
        results = tracker.track(frame)
        frame, results = parser.process_frame(frame, results)

        for id in results:
            x = (
                results[id]["foot_coordinates"]["left"][0]
                + results[id]["foot_coordinates"]["right"][0]
            ) / 2.0
            y = (
                results[id]["foot_coordinates"]["left"][1]
                + results[id]["foot_coordinates"]["right"][1]
            ) / 2.0
            pant_color = results[id]["lower_body_color"]["name"]
            shirt_color = results[id]["torso_color"]["name"]
            confidence = results[id]["confidence"]
            face_bbox = results[id]["face_bbox"]
            bbox = results[id]["bbox"]
            left, top, right, bottom = map(int, bbox)
            f_left, f_top, f_right, f_bottom = map(int, face_bbox)
            face_crop = frame[f_top:f_bottom, f_left:f_right]
            hair_percentage = Hairline(face_crop, bisenet)
            action_id = classify_action(frame[top:bottom, left:right], vit)
            race_pred, age_pred, gender_pred, race_score, age_score, gender_score = (
                predict_age_gender_race(face_crop, fair_face)
            )

            print(f"{'Person'} {id}")
            print(f"{'% Hair'} {hair_percentage}")
            print(f"{'Action'} {actions[action_id]}")
            print(f"{'Race:'} {idx_race[race_pred]} {race_score:.2f}")
            print(f"{'Age: '} {idx_age[age_pred]} {age_score:.2f}")
            print(f"{'Gender: '} {idx_gender[gender_pred]} {gender_score:.2f}")
