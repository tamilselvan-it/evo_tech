import sys
import os
import csv
import numpy as np
from datetime import datetime
from keras.models import load_model
from keras.preprocessing import image

# ✅ Load the trained model
model = load_model("fertile_model.h5")

# ✅ Define image size used during training
img_width, img_height = 150, 150

# ✅ Check if image path is passed
if len(sys.argv) < 2:
    print("❌ Please provide an image path.")
    sys.exit()

image_path = sys.argv[1]

# ✅ Log prediction results to CSV
def log_prediction(image_path, prediction_label, confidence):
    csv_file = "scan_results.csv"
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "image_name", "prediction", "confidence", "image_path"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            os.path.basename(image_path),
            prediction_label,
            f"{confidence:.2f}%",
            image_path
        ])

try:
    # ✅ Load and preprocess the image
    img = image.load_img(image_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize

    # ✅ Make prediction
    prediction = model.predict(img_array)
    class_index = int(prediction[0][0] > 0.5)  # 0 = fertile, 1 = unfertile
    confidence = float(prediction[0][0]) if class_index == 1 else 1 - float(prediction[0][0])

    # ✅ Class mapping
    class_mapping = {0: 'fertile', 1: 'unfertile'}
    predicted_label = class_mapping[class_index]

    # ✅ Show result
    print(f"✅ Prediction: {predicted_label} (confidence: {confidence * 100:.2f}%)")

    # ✅ Log the result
    log_prediction(image_path, predicted_label, confidence * 100)

except FileNotFoundError:
    print(f"❌ File not found: {image_path}")
except Exception as e:
    print(f"❌ Error: {e}")
