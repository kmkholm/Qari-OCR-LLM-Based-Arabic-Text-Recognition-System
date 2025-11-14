# ================== INSTALL ==================
!apt-get install -y poppler-utils fonts-dejavu-core > /dev/null
!pip install -q "transformers>=4.43.0" "qwen_vl_utils" "accelerate>=0.26.0" "peft" "bitsandbytes" pdf2image pillow arabic-reshaper python-bidi fpdf2

# ================== IMPORTS ==================
import os
import torch
from pdf2image import convert_from_path
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from bidi.algorithm import get_display
import arabic_reshaper
from fpdf import FPDF
from google.colab import files

# ================== TOKENS ==================
HF_TOKEN = os.environ.get("HF_TOKEN", "put ur key")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "put ur key")

os.environ["WANDB_DISABLED"] = "true"

auth_kwargs = {}
if HF_TOKEN:
    auth_kwargs["token"] = HF_TOKEN

# ================== LOAD MODEL ==================
model_name = "NAMAA-Space/Qari-OCR-0.1-VL-2B-Instruct"
print("Loading Qari-OCR model...")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    **auth_kwargs
)
processor = AutoProcessor.from_pretrained(model_name, **auth_kwargs)

max_tokens = 2000
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Model loaded. Device:", device)

# ================== UPLOAD FILE(S) ==================
print("Upload PDF and/or image files (jpg/png/jpeg/tif/webp)...")
uploaded = files.upload()
file_paths = list(uploaded.keys())
print("Uploaded:", file_paths)

valid_img_ext = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# ================== Qari-OCR HELPER ==================
def qari_ocr_image_path(src_path: str) -> str:
    """Run Qari-OCR on a single image file path and return plain text."""
    img = Image.open(src_path).convert("RGB")
    img.save(src_path)

    prompt = (
        "Below is the image of one page of an Arabic document. "
        "Return only the exact plain Arabic text as seen on the page. "
        "Do not translate, add, or remove text."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{src_path}"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output_text.strip()

# ================== MAIN OCR LOOP ==================
all_pages_text = []
page_counter = 1

for path in file_paths:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        print(f"\n[PDF] Converting pages to images: {path}")
        pages = convert_from_path(path, dpi=300)
        images_for_file = []
        os.makedirs("pdf_pages", exist_ok=True)
        for idx, page in enumerate(pages, start=1):
            img_path = os.path.join("pdf_pages", f"{os.path.basename(path)}_page_{idx}.png")
            page.save(img_path, "PNG")
            images_for_file.append(img_path)
    elif ext in valid_img_ext:
        print(f"\n[IMG] Using image directly: {path}")
        images_for_file = [path]
    else:
        print(f"\n[SKIP] Unsupported file type: {path}")
        continue

    for idx, img_path in enumerate(images_for_file, start=1):
        print(f"  -> Qari-OCR on {img_path}")
        try:
            text = qari_ocr_image_path(img_path)
        except Exception as e:
            print("     ERROR while running Qari-OCR:", e)
            continue

        if not text.strip():
            print("     (warning: empty text from model)")
            continue

        preview = " ".join(text.split()[:10])
        print(f"     text preview: {preview!r}")

        all_pages_text.append((page_counter, text))
        page_counter += 1

print("\nQari-OCR finished.")

# ================== SAVE TXT ==================
if not all_pages_text:
    print("No text returned from Qari-OCR. Nothing to save.")
else:
    TXT_OUT = "qari_ocr_output.txt"
    with open(TXT_OUT, "w", encoding="utf-8") as f:
        for page_num, text in all_pages_text:
            f.write(f"===== PAGE {page_num} =====\n")
            f.write(text)
            f.write("\n\n")
    print("Saved TXT:", TXT_OUT)

    with open(TXT_OUT, "r", encoding="utf-8") as f:
        print("TXT preview:", f.read(500))

    # ================== SAVE ARABIC PDF (proper shaping) ==================
    PDF_OUT = "qari_ocr_output.pdf"

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.add_font("DejaVu", "", font_path)
    pdf.set_font("DejaVu", size=14)

    usable_width = pdf.w - pdf.l_margin - pdf.r_margin

    first_page = True
    for page_num, text in all_pages_text:
        if not first_page:
            pdf.add_page()
        first_page = False

        header = f"PAGE {page_num}"
        pdf.set_font("DejaVu", size=12)
        pdf.cell(0, 10, header, new_x="LMARGIN", new_y="NEXT", align="L")
        pdf.set_font("DejaVu", size=14)

        for raw_line in text.split("\n"):
            line = raw_line.strip()
            if not line:
                pdf.ln(6)
                continue

            reshaped = arabic_reshaper.reshape(line)
            bidi_text = get_display(reshaped)

            try:
                pdf.multi_cell(usable_width, 8, bidi_text, align="R")
            except Exception as e:
                # Fallback: skip or truncate problematic line
                print("Warning: could not render line in PDF:", e)
                continue

    pdf.output(PDF_OUT)
    print("Saved PDF:", PDF_OUT)

    files.download(TXT_OUT)
    files.download(PDF_OUT)