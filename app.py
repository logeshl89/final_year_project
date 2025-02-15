import logging
import os
import cv2
import random
from flask import Flask, request, jsonify, render_template, send_file, send_from_directory
from roboflow import Roboflow
from fpdf import FPDF
import math
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


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/form')
def form_page():
    return render_template('form.html')

@app.route('/image_form')
def image_form_page():
    return render_template('image_form.html')
@app.route("/uploads/<filename>")
def serve_uploaded_image(filename):
    return send_from_directory(IMAGE_UPLOAD_FOLDER, filename)



IMAGE_UPLOAD_FOLDER = os.path.join(UPLOAD_FOLDER, "images")
os.makedirs(IMAGE_UPLOAD_FOLDER, exist_ok=True)

@app.route("/predict_crops", methods=["POST"])
def predict_crops():
    try:
        if "cropImages" not in request.files:
            return jsonify({"error": "No files uploaded"}), 400

        files = request.files.getlist("cropImages")
        predictions = []

        for file in files:
            filename = os.path.join(IMAGE_UPLOAD_FOLDER, file.filename)
            file.save(filename)  # Save file to the images folder

            # Run inference using Roboflow
            prediction = model.predict(filename, confidence=40, overlap=30)

            # Save the predicted image with annotations
            predicted_filename = os.path.join(
                IMAGE_UPLOAD_FOLDER, file.filename.replace(".jpg", "_predicted.jpg")
            )
            prediction.save(predicted_filename)

            predictions.append({
                "image": f"images/{file.filename}",  # Relative path for displaying
                "predicted_image": f"images/{os.path.basename(predicted_filename)}",
                "result": prediction.json()
            })

        # Render the results page with the original and predicted images
        return render_template("result.html", predictions=predictions)

    except Exception as e:
        return render_template("result.html", error=f"An error occurred: {str(e)}")


def calculate_area_acres(coordinates):

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

class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", style="B", size=12)
        self.cell(200, 10, "Farmer Disaster Report", ln=True, align="C")
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
            img_path = os.path.join(output_folder, f"frame_{i+1}.jpg")
            cv2.imwrite(img_path, frame)
            extracted_images.append(img_path)

    cap.release()
    return extracted_images


def run_inference(images):
    """Runs inference on extracted frames using Roboflow."""
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

            # Extract frames and run inference
            extracted_frames = extract_frames(survey_video_path, UPLOAD_FOLDER, num_frames=3)
            if extracted_frames:
                results, prediction_images = run_inference(extracted_frames)
            else:
                results, prediction_images = [], []
        area = calculate_area_acres(coordinates)
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

        # Add survey images and predictions to the PDF
        if results:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(200, 10, "Survey Images & Predictions", ln=True, align="C")
            pdf.ln(5)

            for i, (original_img, prediction) in enumerate(results):
                pdf.set_font("Arial", "B", 10)
                pdf.cell(200, 10, f"Image {i+1} Analysis", ln=True, align="L")

                # Add images
                pdf.image(original_img, x=30, w=80)
                pdf.ln(5)
                pdf.image(prediction_images[i], x=30, w=80)
                pdf.ln(3)

                # Add prediction data
                pdf.set_font("Arial", "", 10)
                pdf.multi_cell(0, 5, f"Prediction Results: {prediction}")
                pdf.ln(5)

        pdf_filename = os.path.join(UPLOAD_FOLDER, f"report_{farmer_id}.pdf")
        pdf.output(pdf_filename)

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
