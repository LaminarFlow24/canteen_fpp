import io
import joblib
from PIL import Image, ImageDraw, ImageFont
from face_recognition import preprocessing
import os

# Specify local paths for model, input image, and output directory
model_file_path = "model/frames_trained.pkl"  # Local path for the face recognition model
image_file_path = "images/WIN_20241113_16_26_28_Pro.jpg"  # Local path for the input image
output_folder_path = "Output_images"  # Folder to save the processed image

# Ensure the output folder exists
os.makedirs(output_folder_path, exist_ok=True)

# Load the face recognizer model from a local file
def load_face_recogniser_model():
    with open(model_file_path, 'rb') as model_file:
        return joblib.load(model_file)

# Initialize face recognizer model
face_recogniser = load_face_recogniser_model()

# Preprocess function
preprocess = preprocessing.ExifOrientationNormalize()

# Load and preprocess the image
img = Image.open(image_file_path)
img = preprocess(img)
img = img.convert('RGB')  # Convert image to RGB (stripping alpha channel if exists)

# Perform face recognition
faces = face_recogniser(img)

# Draw bounding boxes and labels on the image
draw = ImageDraw.Draw(img)
font = ImageFont.load_default()  # You can load a custom font if necessary

# Display recognition results
print("Recognition Results:")
if faces:
    for idx, face in enumerate(faces):
        print(f"Face {idx + 1}:")
        print(f"Top Prediction: {face.top_prediction.label} (Confidence: {face.top_prediction.confidence:.2f})")
        print(f"Bounding Box: Left: {face.bb.left}, Top: {face.bb.top}, Right: {face.bb.right}, Bottom: {face.bb.bottom}")
        
        # Draw bounding box
        draw.rectangle([face.bb.left, face.bb.top, face.bb.right, face.bb.bottom], outline="red", width=2)
        
        # Prepare label text and its size
        label_text = f"{face.top_prediction.label} ({face.top_prediction.confidence:.2f})"
        text_bbox = draw.textbbox((face.bb.left, face.bb.top - 10), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Draw black rectangle for label background
        draw.rectangle([face.bb.left, face.bb.top - text_height, face.bb.left + text_width, face.bb.top], fill="black")
        
        # Draw white label text
        draw.text((face.bb.left, face.bb.top - text_height), label_text, fill="white", font=font)
else:
    print("No faces detected.")

# Save the processed image in the specified folder
output_image_path = os.path.join(output_folder_path, "processed_image.jpg")
img.save(output_image_path)
print(f"Processed image saved at {output_image_path}")
