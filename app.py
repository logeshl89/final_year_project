import logging
import os
import cv2
import random
from flask import Flask, request, jsonify, render_template, send_file
from roboflow import Roboflow
from fpdf import FPDF
import cv2
import matplotlib.pyplot as plt

import math

def generate_confidence_graph(form_data, video_path, confidence_level):
    # Extract damage reason from form data
    reason = form_data.get("reason", "Other")

    # Compute affected vs non-affected area percentages
    affected_area = confidence_level * 100
    non_affected_area = 100 - affected_area

    # Load video and extract a frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Failed to extract a frame from video.")
        return None

    # Convert BGR to RGB for displaying in Matplotlib
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Labels and values for affected vs non-affected
    labels = ["Affected Area", "Non-Affected Area"]
    values = [affected_area, non_affected_area]
    colors = ["red", "green"]

    bars = ax.bar(labels, values, color=colors)
    ax.set_title(f"Disaster Impact Analysis ({reason})", fontsize=14)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_ylim(0, 100)

    # Add percentage labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.1f}%", ha='center', fontsize=12, fontweight='bold')
    # Overlay extracted video frame in the background
    fig.figimage(frame, xo=50, yo=50, alpha=0.4)  # Position at bottom-left with transparency

    # Save graph
    output_path = "static/confidence_graph.png"
    plt.savefig(output_path)
    plt.close()

    print(f" Graph saved at: {output_path}")

    return output_path


def calculate_area_acres(coordinates):
    # ... (input validation remains the same)

    area_sq_meters = 0.0
    num_points = len(coordinates)

    for i in range(num_points - 1):
        lat1, lon1 = coordinates[i]
        lat2, lon2 = coordinates[(i + 1) % num_points]
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        area_sq_meters += (lon2_rad - lon1_rad) * (math.sin(lat1_rad) + math.sin(lat2_rad))
    EARTH_RADIUS = 6378137.0  # meters
    area_sq_meters = abs(area_sq_meters * EARTH_RADIUS * EARTH_RADIUS / 2)
    acres = area_sq_meters / 4046.86
    print(acres)
    return acres


# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Initialize Roboflow Model
rf = Roboflow(api_key="iW1QmBy39feV54Qtr575")
project = rf.workspace().project("banana-jtjak")
model = project.version(1).model


class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", style="B", size=14)
        self.cell(200, 10, "Farmer Disaster Report", ln=True, align='C')
        self.ln(10)

    def add_section(self, title, text):
        self.set_font("Arial", "B", 12)

        # Define column widths
        col_widths = [50, 140]  # Title column, Text column

        # Add a table row with a border
        self.cell(col_widths[0], 10, title, border=1, align="C")
        self.set_font("Arial", size=10)
        self.multi_cell(col_widths[1], 10, text, border=1)

        # Space after the row
        self.ln(5)

        logging.debug(f"Added section: {title} - {text}")

    def add_image_section(self, img_path):
        self.set_font("Arial", "B", 12)
        self.cell(200, 10, "Survey Image & Prediction", ln=True, align="C")
        self.ln(5)
        # Add original image and prediction
        self.image(img_path, x=30, w=80)
        self.ln(5)
        self.ln(5)
    def add_image_graph(self, img_path):
        self.set_font("Arial", "B", 12)
        self.cell(200, 10, "Graph", ln=True, align="C")
        self.ln(5)
        self.image(img_path, x=30, w=80)
        self.ln(5)
        self.ln(5)



def extract_frames(video_path, output_folder, num_frames=3):
    """Extracts random frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count == 0:
        logging.error("No frames found in video.")
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


@app.route("/")
def form():
    return render_template("index.html")


@app.route("/generate_report", methods=["POST"])
def generate_report():
    try:
        # Retrieve form data
        land_type = request.form.get("land_type")
        farmer_name = request.form.get("farmerName", "N/A")
        farmer_id = request.form.get("farmerId", "N/A")
        contact_number = request.form.get("contactNumber", "N/A")
        reason = request.form.get("reason", "N/A")

        farm_location = request.form.get("farmLocation", "N/A")
        farmer_location = request.form.get("farmerLocation", "N/A")
        plant_growth = request.form.get("plantGrowth", "N/A")
        coordinates = [
            (float(request.form.get("coordinates1", "0").split(",")[0]),
             float(request.form.get("coordinates1", "0").split(",")[1])),
            (float(request.form.get("coordinates2", "0").split(",")[0]),
             float(request.form.get("coordinates2", "0").split(",")[1])),
            (float(request.form.get("coordinates3", "0").split(",")[0]),
             float(request.form.get("coordinates3", "0").split(",")[1])),
            (float(request.form.get("coordinates4", "0").split(",")[0]),
             float(request.form.get("coordinates4", "0").split(",")[1]))
        ]
        # print(coordinates)
        drone_model = request.form.get("droneModel", "N/A")
        drone_number = request.form.get("droneNumber", "N/A")
        pilot_name = request.form.get("pilotName", "N/A")

        # Handle file uploads
        farm_document = request.files.get("farmDocument")
        survey_video = request.files.get("surveyVideo")

        # Save uploaded files
        if farm_document:
            farm_doc_path = os.path.join(UPLOAD_FOLDER, "farm_document.pdf")
            farm_document.save(farm_doc_path)
        if survey_video:
            survey_video_path = os.path.join(UPLOAD_FOLDER, "survey_video.mp4")
            survey_video.save(survey_video_path)
            extracted_frames = extract_frames(survey_video_path, UPLOAD_FOLDER, num_frames=3)
            if extracted_frames:
                results, prediction_images = run_inference(extracted_frames)
            else:
                results, prediction_images = [], []

        area = calculate_area_acres(coordinates)
        #Generating Grph
        form_data = {
            "reason": land_type
        }
        video_path = "uploads/survey_video.mp4"  # Path to uploaded video

        graph_path = generate_confidence_graph(form_data, video_path, .75)
        if graph_path:
            print(f"Graph generated successfully: {graph_path}")
        # Generate PDF
        pdf = PDFReport()
        pdf.add_page()
        pdf.add_section("Farmer Name:", farmer_name)
        pdf.add_section("Farmer ID:", farmer_id)
        pdf.add_section("Contact Number:", contact_number)
        pdf.add_section("Reason for Affected:", reason)
        pdf.add_section("Farm Location:", farm_location)
        pdf.add_section("Farmer Location:", farmer_location)
        pdf.add_section("Growth of the Plant:", plant_growth)

        rate_per_acre = 5000 if land_type.lower() == "rural" else 4500
        compensation_amount = area * rate_per_acre
        pdf.add_section("Compensation Amount:", f"{compensation_amount:,.2f}")
        pdf.add_section("Affected Area (in acres):", f"{float(area):.2f}")
        pdf.add_section("Drone Model Used:", drone_model)
        pdf.add_section("Drone Number:", drone_number)
        pdf.add_section("Pilot Name:", pilot_name)
        # # Add survey images and predictions to the PDF

        if results:
            for i, (original_img, prediction) in enumerate(results):
                pdf.add_image_section(original_img)
        pdf_filename = os.path.join(UPLOAD_FOLDER, f"report_{farmer_id+"_Report"}.pdf")
        pdf.output(pdf_filename)
        pdf.add_image_section(graph_path)
        print("Added")
        return jsonify({"message": "PDF Report Generated", "file": pdf_filename})

    except Exception as e:
        logging.error(f"Error generating report: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route("/download_report/<farmer_id>")
def download_report(farmer_id):
    """Endpoint to download the generated PDF."""
    pdf_filename = os.path.join(UPLOAD_FOLDER, f"report_{farmer_id}.pdf")
    if os.path.exists(pdf_filename):
        return send_file(pdf_filename, as_attachment=True)
    else:
        return jsonify({"error": "Report not found"}), 404


if __name__ == "__main__":
    app.run(debug=True)



