from flask import Flask, request, render_template, send_file
import openai
from io import BytesIO
import os
from PIL import Image
import base64
import requests
from runwayml import RunwayML, TaskFailedError
import glob

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- API Keys ---
OPENAI_API_KEY = "sk-proj-xxxxxxxxx"
RUNWAY_API_KEY = "key_xxxxxxxxxxxxx"

openai.api_key = OPENAI_API_KEY
runway_client = RunwayML(api_key=RUNWAY_API_KEY)

# --- Prompts ---
PROMPT_IMAGE = (
    "Transform this image to realistically reflect Bangkok in the 1960s. "
    "Keep original composition, retro colors, vintage cars, old shop signs, 1960s clothing."
)
PROMPT_VIDEO = "Short 5-second video, gentle camera motion, vintage 1960s street style"

# --- ฟังก์ชันช่วยสร้างชื่อไฟล์เรียงลำดับ ---
def get_next_filename(folder, prefix="BangkokEra", ext=".png"):
    os.makedirs(folder, exist_ok=True)
    files = glob.glob(os.path.join(folder, f"{prefix}*{ext}"))
    if not files:
        return os.path.join(folder, f"{prefix}001{ext}")
    numbers = [int(os.path.splitext(f)[0].split(prefix)[-1]) for f in files]
    next_num = max(numbers) + 1
    return os.path.join(folder, f"{prefix}{next_num:03d}{ext}")

# --- ฟังก์ชันแปลงภาพ OpenAI ---
def convert_image_to_1960s(file):
    allowed_exts = (".png", ".jpg", ".jpeg", ".webp")
    if not file.filename.lower().endswith(allowed_exts):
        raise ValueError("Only PNG, JPG, JPEG, or WebP images are supported.")

    file.seek(0)
    img = Image.open(file)
    if img.mode != "RGB":
        img = img.convert("RGB")

    buffered = BytesIO()
    img.save(buffered, format="PNG")
    buffered.seek(0)

    response = openai.images.edit(
        model="gpt-image-1",
        image=("input.png", buffered, "image/png"),
        prompt=PROMPT_IMAGE,
        size="1024x1024"
    )

    if response.data and response.data[0].b64_json:
        return base64.b64decode(response.data[0].b64_json)
    else:
        raise ValueError("OpenAI did not return valid image data.")

# --- ฟังก์ชันสร้างวิดีโอ Runway ---
def generate_video_from_image(img_bytes, output_path="output.mp4"):
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    task = runway_client.image_to_video.create(
        model="gen4_turbo",
        prompt_image=f"data:image/png;base64,{img_b64}",
        prompt_text=PROMPT_VIDEO,
        ratio="1280:720",
        duration=5
    ).wait_for_task_output()

    if not task.output:
        raise ValueError("Runway did not return a valid video URL.")

    video_url = task.output[0] if isinstance(task.output[0], str) else task.output[0].get("url")
    if not video_url:
        raise ValueError("Runway did not return a valid video URL.")

    r = requests.get(video_url)
    if r.status_code != 200:
        raise ValueError("Failed to download video from Runway.")

    with open(output_path, "wb") as f:
        f.write(r.content)

    return output_path

# --- Flask routes ---
@app.route("/", methods=["GET", "POST"])
def index():
    message = ""
    img_file = None
    video_file = None

    if request.method == "POST":
        if "image" not in request.files:
            message = "No file uploaded."
        else:
            file = request.files["image"]
            if file.filename == "":
                message = "No file selected."
            else:
                try:
                    # แปลงภาพด้วย OpenAI
                    img_bytes = convert_image_to_1960s(file)

                    # กำหนดโฟลเดอร์สำหรับเก็บไฟล์
                    images_folder = os.path.join(app.config['UPLOAD_FOLDER'], "images_database")
                    videos_folder = os.path.join(app.config['UPLOAD_FOLDER'], "videos_database")
                    os.makedirs(images_folder, exist_ok=True)
                    os.makedirs(videos_folder, exist_ok=True)

                    # สร้างชื่อไฟล์อัตโนมัติ
                    output_img_path = get_next_filename(images_folder, ext=".png")
                    with open(output_img_path, "wb") as f:
                        f.write(img_bytes)
                    img_file = output_img_path

                    output_video_path = get_next_filename(videos_folder, ext=".mp4")
                    video_file = generate_video_from_image(img_bytes, output_video_path)

                except Exception as e:
                    message = f"Error: {str(e)}"

    return render_template("index.html", message=message, img_file=img_file, video_file=video_file)

@app.route("/image")
def image():
    images_folder = os.path.join(app.config['UPLOAD_FOLDER'], "images_database")
    latest_image = get_next_filename(images_folder, ext=".png")
    latest_image = os.path.join(images_folder, f"BangkokEra{int(latest_image[-7:-4])-1:03d}.png")
    return send_file(latest_image, mimetype="image/png")

@app.route("/video")
def video():
    videos_folder = os.path.join(app.config['UPLOAD_FOLDER'], "videos_database")
    latest_video = get_next_filename(videos_folder, ext=".mp4")
    latest_video = os.path.join(videos_folder, f"BangkokEra{int(latest_video[-7:-4])-1:03d}.mp4")
    return send_file(latest_video, mimetype="video/mp4")

if __name__ == "__main__":
    app.run(debug=True)