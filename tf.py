from flask import Flask, request, render_template, jsonify
import os
import tempfile
import face_recognition
import numpy as np
import requests
import base64

app = Flask(__name__)

# --- Database: Roll numbers with image URLs ---
students_db = [
    {"roll_no": "23691A3262", "url": "https://drive.google.com/uc?export=download&id=1-asL-UaDvyWwaEvEFS10QYRgT6aS5P5z"},
    {"roll_no": "23691A05G6", "url": "https://drive.google.com/uc?export=download&id=12Lsq_gYfGa0h9gle924BSOZ0pomfArcq"},
    {"roll_no": "23691A05J5", "url": "https://drive.google.com/uc?export=download&id=1PtQqqgn0dNprzPE5kYg_2Zl5x2PwgjV5"},
    {"roll_no": "23691A3294", "url": "https://drive.google.com/uc?export=download&id=17BgeX98YvtuSbUgNFqaVNmJYwUHv3EFK"},
    {"roll_no": "23691A05G3", "url": "https://drive.google.com/uc?export=download&id=1kd4HNCxMpiVeYmTfjoNA1m9s5f-ndUn9"},
    {"roll_no": "23691A3295", "url": "https://drive.google.com/uc?export=download&id=14U4DGVG8_4MJtTS0VtdCft5nqzu7li_U"}
]

# --- Precompute encodings for database students ---
print("🔄 Loading and encoding student database images...")
for student in students_db:
    try:
        response = requests.get(student["url"])
        if response.status_code == 200:
            tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            tmpf.close()
            with open(tmpf.name, "wb") as f:
                f.write(response.content)
            image = face_recognition.load_image_file(tmpf.name)
            encodings = face_recognition.face_encodings(image)
            os.remove(tmpf.name)
            student["encoding"] = encodings[0] if encodings else None
    except Exception as e:
        print(f"⚠️ Failed to load {student['roll_no']}: {e}")
print("✅ Student encodings loaded!")

# --- Helper Functions ---
def save_temp_image(content):
    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tmpf.close()
    with open(tmpf.name, "wb") as f:
        f.write(content)
    return tmpf.name

def compare_face_with_db(face_encoding, tolerance=0.5):
    """Compare a single face encoding with all students in DB"""
    matches = []
    for student in students_db:
        if student.get("encoding") is None:
            continue
        dist = np.linalg.norm(face_encoding - student["encoding"])
        if dist <= tolerance:  # Lower distance = closer match
            matches.append({
                "roll_no": student["roll_no"],
                "matched_url": student["url"],
                "distance": round(float(dist), 4)
            })
    return matches

# --- Routes ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    tmp_path = None

    if "image" in request.files:
        file = request.files["image"]
        tmp_path = save_temp_image(file.read())
    elif "image_base64" in request.form:
        image_base64 = request.form["image_base64"]
        if image_base64.startswith("data:image"):
            image_base64 = image_base64.split(",")[1]
        img_data = base64.b64decode(image_base64)
        tmp_path = save_temp_image(img_data)
    else:
        return jsonify({"error": "No image provided"}), 400

    uploaded_image = face_recognition.load_image_file(tmp_path)
    face_locations = face_recognition.face_locations(uploaded_image)
    encodings = face_recognition.face_encodings(uploaded_image, face_locations)
    os.remove(tmp_path)

    if not encodings:
        return jsonify({"error": "No faces detected in uploaded image"}), 400

    results = []
    for idx, (encoding, location) in enumerate(zip(encodings, face_locations)):
        matches = compare_face_with_db(encoding)
        results.append({
            "face_index": idx,
            "location": location,  # (top, right, bottom, left)
            "matches": matches
        })

    return jsonify({
        "status": "Processed",
        "faces_detected": len(encodings),
        "results": results
    })

if __name__ == "__main__":
    app.run(debug=True)
