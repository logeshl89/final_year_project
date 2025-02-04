import cv2
import random
import os
from roboflow import Roboflow
from fpdf import FPDF

# Initialize Roboflow Model
rf = Roboflow(api_key="iW1QmBy39feV54Qtr575")
project = rf.workspace().project("banana-jtjak")
model = project.version(1).model


# Function to extract frames from video
def extract_frames(video_path, output_folder, num_frames=3):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count == 0:
        print("Error: No frames found in video.")
        return []

    selected_frames = random.sample(range(0, frame_count), min(num_frames, frame_count))
    extracted_images = []

    for i, frame_no in enumerate(selected_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if ret:
            img_path = os.path.join(output_folder, f"frame_{i + 1}.jpg")
            cv2.imwrite(img_path, frame)
            extracted_images.append(img_path)

    cap.release()
    return extracted_images


# Function to perform inference
def run_inference(images):
    prediction_images = []
    results = []

    for img in images:
        prediction = model.predict(img, confidence=40, overlap=30)
        prediction_img = img.replace(".jpg", "_prediction.jpg")
        model.predict(img, confidence=40, overlap=30).save(prediction_img)

        results.append((img, prediction.json()))
        prediction_images.append(prediction_img)

    return results, prediction_images


# Function to generate PDF
def generate_pdf(results, prediction_images, pdf_filename):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Farmer Disaster Report", ln=True, align="C")

    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 10, "Survey Images & Predictions", ln=True, align="C")
    pdf.ln(10)

    for i, (original_img, prediction) in enumerate(results):
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, f"Image {i + 1} Analysis", ln=True, align="L")

        # Add original image
        pdf.image(original_img, x=30, w=80)
        pdf.ln(5)

        # Add prediction image
        pdf.image(prediction_images[i], x=30, w=80)
        pdf.ln(5)

        # Add prediction data
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 5, f"Prediction Results: {prediction}")
        pdf.ln(10)

    pdf.output(pdf_filename)


# Main Execution
video_path = "File.mp4"  # Change this to your video file path
output_folder = "frames"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Extract frames
extracted_frames = extract_frames(video_path, output_folder, num_frames=3)

if extracted_frames:
    # Run inference
    results, prediction_images = run_inference(extracted_frames)

    # Generate PDF report
    pdf_filename = "survey_report.pdf"
    generate_pdf(results, prediction_images, pdf_filename)
    print(f"PDF report generated: {pdf_filename}")
else:
    print("No frames extracted, unable to proceed.")
