# Automatic Accident Detection System
# YOLOv8 + OpenCV + Telegram Bot alerts (Google Colab version)
# Converted from the project code notes into a single script, section by section.


# ============================================================
# 1. Install Required Libraries
# ============================================================
# !pip install ultralytics
# !pip install opencv-python
# !pip install requests
# !pip install torch torchvision
# !pip install pillow

# ============================================================
# 2. Import Required Modules
# ============================================================
from ultralytics import YOLO
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import os

# ============================================================
# 3. Mount Google Drive
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

# ============================================================
# 4. Dataset Folder Paths
# ============================================================
base_path = "/content/drive/MyDrive/Accident_Project/final_dataset"

train_img = f"{base_path}/train/images"
valid_img = f"{base_path}/valid/images"
test_img  = f"{base_path}/test/images"

train_lbl = f"{base_path}/train/labels"
valid_lbl = f"{base_path}/valid/labels"
test_lbl  = f"{base_path}/test/labels"

# ============================================================
# 5. Create Label Folders
# ============================================================
for path in [train_lbl, valid_lbl, test_lbl]:
    os.makedirs(path, exist_ok=True)

print("Folders Created")

# ============================================================
# 6. Rename Dataset Images
# ============================================================
def rename_images(folder):

    acc_count = 1
    non_count = 1

    for file in os.listdir(folder):

        old_path = os.path.join(folder, file)

        if "acc" in file.lower():

            new_name = f"acci_{acc_count}.jpg"
            acc_count += 1

        else:

            new_name = f"non_acc_{non_count}.jpg"
            non_count += 1

        new_path = os.path.join(folder, new_name)

        os.rename(old_path, new_path)

    print("Renaming Completed")

rename_images(train_img)

# ============================================================
# 7. Create YOLO Labels
# ============================================================
def create_labels(img_folder, lbl_folder):

    for file in os.listdir(img_folder):

        if not file.endswith(".jpg"):
            continue

        label_path = os.path.join(
            lbl_folder,
            file.replace(".jpg", ".txt")
        )

        if "acci_" in file:

            label = "0 0.5 0.5 1 1"

        else:

            label = "1 0.5 0.5 1 1"

        with open(label_path, "w") as f:
            f.write(label)

    print("Labels Generated")

create_labels(train_img, train_lbl)
create_labels(valid_img, valid_lbl)
create_labels(test_img, test_lbl)

# ============================================================
# 8. Resize Dataset Images
# ============================================================
from PIL import Image

IMG_SIZE = 640

splits = ["train", "valid", "test"]

for split in splits:

    folder = f"{base_path}/{split}/images"

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        try:

            img = Image.open(path)
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img.save(path)

        except:
            print("Error:", file)

print("Resizing Completed")

# ============================================================
# 9. Create data.yaml File
# ============================================================
data_yaml = """
path: /content/drive/MyDrive/Accident_Project/final_dataset

train: train/images
val: valid/images
test: test/images

nc: 2
names: ['accident', 'non_accident']
"""

with open(
    "/content/drive/MyDrive/Accident_Project/final_dataset/data.yaml",
    "w"
) as f:

    f.write(data_yaml)

print("data.yaml Created")

# ============================================================
# 10. Train YOLOv8 Model
# ============================================================
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="/content/drive/MyDrive/Accident_Project/final_dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="accident_model",
    project="/content/drive/MyDrive/Accident_Project"
)

# ============================================================
# 11. Validate Model
# ============================================================
metrics = model.val()

print(metrics)

# ============================================================
# 12. Test Single Image
# ============================================================
results = model.predict(
    source="/content/drive/MyDrive/test.jpg",
    conf=0.5,
    show=True
)

# ============================================================
# 13. Telegram Bot Configuration
# ============================================================
BOT_TOKEN = "YOUR_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"

# ============================================================
# 14. Send Telegram Alert Message
# ============================================================
def send_telegram_alert():

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    data = {
        "chat_id": CHAT_ID,
        "text": "🚨 ACCIDENT DETECTED!"
    }

    requests.post(url, data=data)

    print("Telegram Alert Sent")

# ============================================================
# 15. Send Accident Image to Telegram
# ============================================================
def send_image(frame):

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"

    _, img_encoded = cv2.imencode('.jpg', frame)

    files = {
        "photo": ("image.jpg", img_encoded.tobytes())
    }

    data = {
        "chat_id": CHAT_ID
    }

    requests.post(url, files=files, data=data)

    print("Image Sent")

# ============================================================
# 16. Load Accident Detection Model
# ============================================================
acc_model = YOLO("best.pt")

# ============================================================
# 17. Load Vehicle Detection Model
# ============================================================
veh_model = YOLO("yolov8n.pt")

# ============================================================
# 18. Detection Parameters
# ============================================================
CONF_ACCIDENT = 0.55
CONF_VEHICLE = 0.5

VEHICLE_CLASSES = [2, 3, 5, 7]

FRAME_COUNT = 0
CONFIRM_FRAMES = 5

ALERT_SENT = False

# ============================================================
# 19. Collision Detection Function
# ============================================================
def check_overlap(box1, box2):

    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2

    if (x1 < x2b and x2 > x1b and
        y1 < y2b and y2 > y1b):

        return True

    return False

# ============================================================
# 20. Complete Final Accident Detection System
# ============================================================
from ultralytics import YOLO
import cv2
import requests

BOT_TOKEN = "YOUR_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"

def send_telegram_alert():

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    data = {
        "chat_id": CHAT_ID,
        "text": "🚨 ACCIDENT DETECTED!"
    }

    requests.post(url, data=data)

def send_image(frame):

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"

    _, img_encoded = cv2.imencode('.jpg', frame)

    files = {
        "photo": ("image.jpg", img_encoded.tobytes())
    }

    data = {
        "chat_id": CHAT_ID
    }

    requests.post(url, files=files, data=data)

acc_model = YOLO("best.pt")
veh_model = YOLO("yolov8n.pt")

CONF_ACCIDENT = 0.55
CONF_VEHICLE = 0.5

VEHICLE_CLASSES = [2,3,5,7]

FRAME_COUNT = 0
CONFIRM_FRAMES = 5

ALERT_SENT = False

def check_overlap(box1, box2):

    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2

    if (x1 < x2b and x2 > x1b and
        y1 < y2b and y2 > y1b):

        return True

    return False

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    accident_detected = False
    collision_detected = False

    vehicle_boxes = []

    acc_results = acc_model(frame)

    for r in acc_results:

        for box in r.boxes:

            conf = float(box.conf[0])
            cls = int(box.cls[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            area = (x2 - x1) * (y2 - y1)

            if cls == 0 and conf > CONF_ACCIDENT and area > 8000:

                accident_detected = True

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0,0,255),
                    3
                )

                cv2.putText(
                    frame,
                    "Accident",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,0,255),
                    2
                )

    veh_results = veh_model(frame)

    for r in veh_results:

        for box in r.boxes:

            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if cls in VEHICLE_CLASSES and conf > CONF_VEHICLE:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                vehicle_boxes.append([x1,y1,x2,y2])

                cv2.rectangle(
                    frame,
                    (x1,y1),
                    (x2,y2),
                    (0,255,0),
                    2
                )

                cv2.putText(
                    frame,
                    "Vehicle",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2
                )

    for i in range(len(vehicle_boxes)):

        for j in range(i+1, len(vehicle_boxes)):

            if check_overlap(
                vehicle_boxes[i],
                vehicle_boxes[j]
            ):

                collision_detected = True

    if accident_detected or collision_detected:

        FRAME_COUNT += 1

    else:

        FRAME_COUNT = 0
        ALERT_SENT = False

    if FRAME_COUNT >= CONFIRM_FRAMES:

        cv2.putText(
            frame,
            "CONFIRMED ACCIDENT!",
            (50,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            3
        )

        if not ALERT_SENT:

            send_telegram_alert()

            send_image(frame)

            ALERT_SENT = True

    cv2.imshow("Accident Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# ============================================================
# 21. CNN Model
# ============================================================
class AccidentCNN(nn.Module):

    def __init__(self):

        super(AccidentCNN, self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(

            nn.Linear(32*56*56,128),
            nn.ReLU(),

            nn.Linear(128,2)
        )

    def forward(self,x):

        x = self.conv(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x

# ============================================================
# 22. SPP Layer
# ============================================================
class SPP(nn.Module):

    def __init__(self):

        super(SPP,self).__init__()

    def forward(self,x):

        p1 = F.max_pool2d(x,5,1,2)
        p2 = F.max_pool2d(x,9,1,4)
        p3 = F.max_pool2d(x,13,1,6)

        out = torch.cat([x,p1,p2,p3],dim=1)

        return out

# ============================================================
# 23. SPPF Layer
# ============================================================
class SPPF(nn.Module):

    def __init__(self):

        super(SPPF,self).__init__()

    def forward(self,x):

        y1 = F.max_pool2d(x,5,1,2)
        y2 = F.max_pool2d(y1,5,1,2)
        y3 = F.max_pool2d(y2,5,1,2)

        return torch.cat([x,y1,y2,y3],1)
