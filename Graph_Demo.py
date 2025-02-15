


import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_confidence_graph(form_data, video_path, confidence_level):
    """
    Generates a bar graph representing the affected area percentage based on model confidence.

    Args:
        form_data (dict): Contains form inputs including disaster details.
        video_path (str): Path to the uploaded survey video file.
        confidence_level (float): Confidence level (0 to 1) of affected area from the model.

    Returns:
        str: Path to the saved graph image.
    """

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
        print("❌ Failed to extract a frame from video.")
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
      # Position at bottom-left with transparency

    # Save graph
    output_path = "static/confidence_graph.png"
    plt.savefig(output_path)
    plt.close()

    print(f"✅ Graph saved at: {output_path}")

    return output_path

form_data = {
    "reason": "Flood"
}
video_path = "uploads/survey_video.mp4"  # Path to uploaded video

graph_path = generate_confidence_graph(form_data, video_path,.75)
if graph_path:
    print(f"Graph generated successfully: {graph_path}")
