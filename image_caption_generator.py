import base64
import io
import os
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import requests
from PIL import Image, ImageDraw, ImageFont, ImageGrab, ImageOps, ImageStat, ImageTk

# ---------------------------------------------------------------
# Settings
# ---------------------------------------------------------------
GROQ_MODEL = "qwen/qwen3.8-27b"          # Groq vision model (replaced automatically if Groq removes it)
GEMINI_MODEL = "gemini-flash-latest"      # Gemini vision model
BLIP_MODEL = "Salesforce/blip-image-captioning-base"   # Hugging Face model that runs on this PC
MAX_SIDE = 768                            # image is shrunk to this before sending (smaller = faster upload)

ONLINE, OFFLINE = "Qwen 27B (online)", "BLIP (offline)"   # the two caption models to choose from

GROQ_URL = "https://api.groq.com/openai/v1"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models"

on_wait = None      # the window sets this to show "waiting..." when the free limit is reached


def caption_prompt(length="Short"):
    if length == "Detailed":
        return ("Describe this image in detail in 3 sentences: the main objects, what is happening, "
                "the colors and the place. Reply with only the description.")
    return ("Describe this image in ONE short, simple sentence of at most 12 words, "
            "like: 'A dog is sitting on grass.' Reply with only that sentence.")


def prediction_prompt():
    return ("Look at this image and write ONE caption of at most 25 words that says what you see AND "
            "predicts what is going on: what the main person or animal might be thinking, feeling or about "
            "to do, judging from clues like the face, pose, hands, objects and place. Use words like 'seems', "
            "'probably' or 'might', like: 'A boy sitting on a chair seems lost in thought, probably worried "
            "about his exam.' If there is no person or animal, predict what might happen next. "
            "Reply with only the caption.")


def question_prompt(question):
    return ("Look carefully at this image and answer the question correctly in 1 or 2 short sentences, "
            "in English. If the question asks you to count, count each object one by one.\n"
            f"Question: {question}")


class CaptionError(Exception):
    """An error with a message that is easy for the user to understand."""


class ModelNotFound(CaptionError):
    """The AI model name is wrong or the model was removed."""


class RateLimited(CaptionError):
    """The free usage limit is used up for now (per minute, or per day)."""
    def __init__(self, wait):
        how_long = f"{int(wait) + 1} seconds" if wait < 90 else f"{round(wait / 60)} minutes"
        super().__init__(f"Free usage limit reached. Please wait about {how_long} and try again.")
        self.wait = wait


# ---------------------------------------------------------------
# 1) Load API keys from .env
# ---------------------------------------------------------------
def load_env():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# ---------------------------------------------------------------
# 2) Prepare the image
# ---------------------------------------------------------------
def load_image(path):
    """Open an image and rotate phone photos upright."""
    with Image.open(path) as img:
        return to_rgb(ImageOps.exif_transpose(img))     # phone photos store rotation in EXIF data


def to_rgb(img):
    """Make an RGB image, with transparent parts put on white."""
    if "A" in img.getbands() or "transparency" in img.info:
        img = img.convert("RGBA")
        background = Image.new("RGB", img.size, "white")
        background.paste(img, mask=img.getchannel("A"))   # otherwise transparent areas turn black
        return background
    return img.convert("RGB")


def image_to_base64(img):
    small = img.copy()
    small.thumbnail((MAX_SIDE, MAX_SIDE))
    buffer = io.BytesIO()
    small.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def clean_text(text):
    if "</think>" in text:                          # remove hidden reasoning, if any
        text = text.split("</think>")[-1]
    return " ".join(text.replace("**", "").split()).strip("\"'* ")


# ---------------------------------------------------------------
# 3) Ask the AI about the image
# ---------------------------------------------------------------
def send_request(method, url, headers, payload=None):
    """Send a request to the AI and turn common failures into clear messages."""
    try:
        response = requests.request(method, url, headers=headers, json=payload,
                                    timeout=(10, 30))   # 10 s to connect, 30 s to answer
    except requests.exceptions.Timeout:
        raise CaptionError("The server took too long to answer. Try again.")
    except requests.exceptions.RequestException:
        raise CaptionError("Could not connect to the server. Check your internet connection.")

    if response.ok:
        return response.json()

    if response.status_code == 429:                 # free limit used up; server says how long to wait
        try:
            wait = float(response.headers.get("retry-after", 10))
        except ValueError:
            wait = 10
        raise RateLimited(wait)

    try:
        error = response.json()["error"]
        detail, code = error.get("message", ""), str(error.get("code", ""))
    except Exception:
        detail, code = response.text[:200], ""
    if response.status_code in (401, 403):
        raise CaptionError(f"API key is wrong or not allowed. Check the .env file. ({detail})")
    if response.status_code == 404 or "model" in code:
        raise ModelNotFound(f"Model not available. ({detail})")
    raise CaptionError(f"Server error {response.status_code}: {detail}")


def find_groq_vision_model(headers):
    """Ask Groq which models it has now and pick one that can see images."""
    models = send_request("GET", f"{GROQ_URL}/models", headers)["data"]
    for m in models:
        if m.get("active") and "image" in m.get("input_modalities", []):
            return m["id"]
    raise CaptionError("Groq currently has no model that can read images.")


def groq_payload(model, b64_image, prompt, max_tokens, turn_off_thinking):
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
            ],
        }],
        "temperature": 0.3,
        "max_completion_tokens": max_tokens if turn_off_thinking else max_tokens + 1000,
    }
    if turn_off_thinking:
        payload["reasoning_effort"] = "none"        # answer directly instead of "thinking" first (faster)
    return payload


def ask_groq(b64_image, prompt, max_tokens):
    global GROQ_MODEL
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise CaptionError("GROQ_API_KEY not found in .env")
    headers = {"Authorization": f"Bearer {key}"}
    url = f"{GROQ_URL}/chat/completions"

    try:
        data = send_request("POST", url, headers, groq_payload(GROQ_MODEL, b64_image, prompt, max_tokens, True))
    except ModelNotFound:                           # Groq removes old models often, so find a new one
        GROQ_MODEL = find_groq_vision_model(headers)
        data = send_request("POST", url, headers, groq_payload(GROQ_MODEL, b64_image, prompt, max_tokens, False))
    return data["choices"][0]["message"]["content"] or ""


def ask_gemini(b64_image, prompt, max_tokens):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise CaptionError("GEMINI_API_KEY not found in .env")

    data = send_request(                            # (Gemini decides the answer length itself)
        "POST",
        f"{GEMINI_URL}/{GEMINI_MODEL}:generateContent",
        {"x-goog-api-key": key},
        {"contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/jpeg", "data": b64_image}},
        ]}]},
    )
    if not data.get("candidates"):
        raise CaptionError("Gemini did not describe this image.")
    parts = data["candidates"][0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts)


def ask_ai(img, prompt, max_tokens):
    """Send the image and the prompt to the AI. Returns (answer, name of the AI that answered)."""
    b64_image = image_to_base64(img)
    for attempt in range(2):
        errors, wait = [], None
        # if Groq's free limit is used up, Gemini answers straight away (it has its own limit)
        for name, func in [("Groq", ask_groq), ("Gemini", ask_gemini)]:
            try:
                answer = clean_text(func(b64_image, prompt, max_tokens))
                if answer:
                    return answer, name
                errors.append(f"{name}: gave an empty answer.")
            except RateLimited as e:
                wait = e.wait if wait is None else min(wait, e.wait)
                errors.append(f"{name}: {e}")
            except Exception as e:
                errors.append(f"{name}: {e}")

        if attempt == 0 and wait is not None and wait <= 20:   # every AI is busy: wait a little, try once more
            if on_wait:
                on_wait(wait)
            time.sleep(wait)
        else:
            break
    raise CaptionError("\n\n".join(errors))


def generate_caption(img, length="Short", model=ONLINE):
    """Returns (caption, which AI made it)."""
    if model == OFFLINE:
        return caption_with_blip(img), OFFLINE
    return ask_ai(img, caption_prompt(length), 600 if length == "Detailed" else 150)


def ask_question(img, question):
    """Returns (answer, which AI made it)."""
    return ask_ai(img, question_prompt(question), 300)


def predict_caption(img):
    """Caption Generator: returns (prediction caption, which AI made it). Needs the online model."""
    return ask_ai(img, prediction_prompt(), 150)


# ---------------------------------------------------------------
# 4) Offline AI: the BLIP model from Hugging Face, running on this PC
# ---------------------------------------------------------------
blip = None             # (processor, model), loaded the first time BLIP is needed
blip_lock = threading.Lock()


def load_blip():
    """Load BLIP from this PC. The first time, it is downloaded from Hugging Face (~1 GB)."""
    try:
        from transformers import BlipForConditionalGeneration, BlipProcessor
    except ImportError:
        raise CaptionError("The offline model needs:  pip install torch transformers")
    try:                                            # already downloaded: works without internet
        return (BlipProcessor.from_pretrained(BLIP_MODEL, local_files_only=True),
                BlipForConditionalGeneration.from_pretrained(BLIP_MODEL, local_files_only=True))
    except OSError:
        pass
    try:                                            # first time: download it
        return (BlipProcessor.from_pretrained(BLIP_MODEL),
                BlipForConditionalGeneration.from_pretrained(BLIP_MODEL))
    except Exception:
        raise CaptionError("Could not download the BLIP model. Connect to the internet once and try again.")


def get_blip():
    """Load BLIP only once (if two threads ask at the same time, the second one waits)."""
    global blip
    with blip_lock:
        if blip is None:
            blip = load_blip()
    return blip


def caption_with_blip(img):
    """BLIP: a Vision Transformer turns the image into features, then a text decoder writes the caption."""
    processor, model = get_blip()
    inputs = processor(images=img, return_tensors="pt")     # resize to 384 x 384 and normalize the pixels
    output = model.generate(**inputs, max_new_tokens=20)    # picks the most likely word, one at a time
    caption = processor.decode(output[0], skip_special_tokens=True).strip()
    return caption[:1].upper() + caption[1:] + "."          # "a dog on grass" -> "A dog on grass."


# ---------------------------------------------------------------
# 5) Image analysis (Digital Image Processing)
# ---------------------------------------------------------------
def analyze_image(img):
    """Measure brightness, contrast, main colors and the RGB histogram of an image."""
    small = img.copy()
    small.thumbnail((400, 400))                     # a small copy is faster and gives the same statistics
    gray = ImageStat.Stat(small.convert("L"))       # grayscale: one brightness value (0-255) per pixel

    brightness = gray.mean[0] / 255 * 100           # average pixel value, as %
    contrast = gray.stddev[0] / 127.5 * 100         # how spread out the pixel values are, as %

    # main colors: reduce the image to 5 colors (color quantization) and count the pixels of each.
    # The octree method keeps small but different colors (like a red apple) instead of mixing them away.
    reduced = small.quantize(colors=5, method=Image.Quantize.FASTOCTREE)
    palette = reduced.getpalette()
    total = small.width * small.height
    colors = [(tuple(palette[i * 3:i * 3 + 3]), count / total * 100)
              for count, i in sorted(reduced.getcolors(), reverse=True)]

    return {
        "size": img.size,
        "brightness": brightness,
        "brightness_word": "Dark" if brightness < 35 else "Bright" if brightness > 70 else "Normal",
        "contrast": contrast,
        "contrast_word": "Low" if contrast < 25 else "High" if contrast > 55 else "Normal",
        "colors": colors,
        "histogram": small.histogram(),             # 768 numbers: 256 for red, then green, then blue
    }


# ---------------------------------------------------------------
# 6) Save the image with the caption written under it
# ---------------------------------------------------------------
def load_font(size):
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


def add_caption_to_image(img, caption):
    """Return a copy of the image with a white strip under it holding the caption."""
    font = load_font(max(18, img.width // 28))
    margin = font.size
    max_width = img.width - 2 * margin

    lines, line = [], ""                            # split the caption into lines that fit the width
    for word in caption.split():
        test = f"{line} {word}".strip()
        if line and font.getlength(test) > max_width:
            lines.append(line)
            line = word
        else:
            line = test
    lines.append(line)

    line_height = int(font.size * 1.5)
    result = Image.new("RGB", (img.width, img.height + len(lines) * line_height + 2 * margin), "white")
    result.paste(img, (0, 0))
    draw = ImageDraw.Draw(result)
    y = img.height + margin
    for line in lines:
        draw.text(((img.width - font.getlength(line)) / 2, y), line, font=font, fill="black")
        y += line_height
    return result


# ---------------------------------------------------------------
# 7) Read text aloud (uses the voice built into Windows)
# ---------------------------------------------------------------
def speak(text):
    """Start reading text aloud in the background. Returns the process (call .terminate() to stop)."""
    if sys.platform != "win32":
        raise CaptionError("Reading aloud only works on Windows.")
    command = ("Add-Type -AssemblyName System.Speech; "
               "(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak($env:SPEAK_TEXT)")
    return subprocess.Popen(["powershell", "-NoProfile", "-Command", command],
                            env={**os.environ, "SPEAK_TEXT": text}, creationflags=subprocess.CREATE_NO_WINDOW)


# ---------------------------------------------------------------
# 8) Window helpers and colors
# ---------------------------------------------------------------
BG, CARD, BOX, BORDER = "#f1f5f9", "white", "#f8fafc", "#e2e8f0"   # page, cards, text boxes, thin lines
TEXT, MUTED, LIGHT = "#0f172a", "#64748b", "#94a3b8"                # dark, gray and light gray text
ACCENT, ACCENT_DARK, ACCENT_SOFT = "#4f46e5", "#4338ca", "#eef2ff"  # indigo, the main color of the app
RED, GREEN, AMBER = "#dc2626", "#16a34a", "#f59e0b"
FONT, BOLD = "Segoe UI", "Segoe UI Semibold"
BUTTONS = {     # style: (color, color under the mouse, text color, color when disabled, text color when disabled)
    "primary": (ACCENT, ACCENT_DARK, "white", "#c7d2fe", "white"),
    "soft": (ACCENT_SOFT, "#e0e7ff", ACCENT_DARK, "#f1f5f9", LIGHT),
}
PREVIEW_W, PREVIEW_H = 480, 310
TAB_W, HIST_H = 400, 150        # width of the boxes inside the tabs, height of the histogram


def make_logo(size, back, front):
    """Draw the app logo with Pillow: a sun over two mountains, on a rounded square."""
    big = size * 4                                  # draw it 4x bigger, then shrink it: smooth edges
    u = big / 20                                    # one unit = 1/20 of the logo
    logo = Image.new("RGBA", (big, big))
    draw = ImageDraw.Draw(logo)
    draw.rounded_rectangle((0, 0, big - 1, big - 1), radius=5 * u, fill=back)
    draw.ellipse((12 * u, 4 * u, 16 * u, 8 * u), fill=AMBER)                       # sun
    draw.polygon([(3 * u, 16 * u), (8 * u, 8 * u), (11.5 * u, 13 * u), (13.5 * u, 10.5 * u), (17 * u, 16 * u)],
                 fill=front)                                                         # mountains
    return logo.resize((size, size), Image.Resampling.LANCZOS)


def make_card(parent, padding=16):
    """A white box with a thin gray border."""
    return tk.Frame(parent, bg=CARD, padx=padding, pady=padding, highlightthickness=1, highlightbackground=BORDER)


def make_button(parent, text, command, style="primary", width=0, small=False):
    color, hover, fg, off_color, off_fg = BUTTONS[style]
    button = tk.Button(parent, text=text, command=command, width=width, font=(FONT, 9) if small else (BOLD, 10),
                       bg=color, fg=fg, activebackground=hover, activeforeground=fg, disabledforeground=off_fg,
                       relief="flat", bd=0, highlightthickness=0, cursor="hand2",
                       padx=10 if small else 14, pady=3 if small else 6)
    button.colors = (color, off_color)              # remembered, so the button can turn pale and back
    button.label = text                             # remembered, so "Stop" can turn back into "Speak"

    def hover_effect(inside):                       # a darker color while the mouse is over the button
        if str(button["state"]) == "normal":
            button.configure(bg=hover if inside else color)
    button.bind("<Enter>", lambda e: hover_effect(True))
    button.bind("<Leave>", lambda e: hover_effect(False))
    button.pack(side="left", padx=(0, 8))
    return button


def set_enabled(button, enabled):
    color, off_color = button.colors
    button.configure(state="normal" if enabled else "disabled", bg=color if enabled else off_color,
                     cursor="hand2" if enabled else "arrow")


def make_choice(parent, variable, values, command):
    """Joined buttons to pick one of a few values (the picked one is colored). Calls command() on a new pick."""
    frame = tk.Frame(parent, bg=BORDER, padx=1, pady=1)    # the gray frame shows as a thin border
    frame.enabled, frame.buttons = True, {}

    def pick(value):
        if frame.enabled and variable.get() != value:
            variable.set(value)
            command()

    def hover(value, inside):                       # light color under the mouse (not on the picked one)
        if frame.enabled and variable.get() != value:
            frame.buttons[value].configure(bg=ACCENT_SOFT if inside else CARD)

    def show(*args):                                # color the buttons (also runs when the variable changes)
        for value, button in frame.buttons.items():
            picked = variable.get() == value
            if picked:
                bg, fg = (ACCENT if frame.enabled else "#a5b4fc"), "white"
            else:
                bg, fg = CARD, (MUTED if frame.enabled else LIGHT)
            button.configure(bg=bg, fg=fg, activebackground=bg if picked else ACCENT_SOFT, activeforeground=fg,
                             disabledforeground=fg, state="normal" if frame.enabled else "disabled",
                             cursor="hand2" if frame.enabled else "arrow")

    for value in values:
        button = tk.Button(frame, text=value, command=lambda v=value: pick(v), font=(BOLD, 9),
                           relief="flat", bd=0, highlightthickness=0, padx=12, pady=4)
        button.bind("<Enter>", lambda e, v=value: hover(v, True))
        button.bind("<Leave>", lambda e, v=value: hover(v, False))
        button.pack(side="left", padx=(1 if frame.buttons else 0, 0))
        frame.buttons[value] = button
    frame.show = show
    variable.trace_add("write", show)
    show()
    return frame


def set_choice_enabled(frame, enabled):
    frame.enabled = enabled
    frame.show()


def make_text_box(parent, width, height, font_size):
    """A read-only text area with a fixed size in pixels. Its scrollbar shows only when the text is too long."""
    frame = tk.Frame(parent, width=width, height=height, bg=BOX, highlightthickness=1, highlightbackground=BORDER)
    frame.pack_propagate(False)
    box = tk.Text(frame, wrap="word", font=(FONT, font_size), relief="flat", bd=0, highlightthickness=0,
                  bg=BOX, fg=TEXT, padx=14, pady=12, spacing2=4, selectbackground="#c7d2fe",
                  selectforeground=TEXT, inactiveselectbackground="#c7d2fe", state="disabled", cursor="arrow")
    scrollbar = ttk.Scrollbar(frame, command=box.yview)

    def scroll(first, last):
        if float(first) <= 0 and float(last) >= 1:
            scrollbar.pack_forget()                 # all the text fits: no scrollbar needed
        elif not scrollbar.winfo_manager():
            scrollbar.pack(side="right", fill="y", before=box)
        scrollbar.set(first, last)

    box.configure(yscrollcommand=scroll)
    box.pack(side="left", fill="both", expand=True)
    return frame, box


def draw_meter(canvas, percent):
    """A rounded bar filled up to the given percent."""
    canvas.delete("all")
    width, y = int(canvas["width"]), int(canvas["height"]) / 2
    canvas.create_line(5, y, width - 5, y, width=8, capstyle="round", fill=BORDER)
    if percent > 0:
        canvas.create_line(5, y, 5 + (width - 10) * min(percent, 100) / 100, y,
                           width=8, capstyle="round", fill=ACCENT)


def draw_histogram(histogram, width, height):
    """Draw the RGB histogram as a picture: a see-through filled curve for each color."""
    chart = Image.new("RGBA", (width, height), CARD)
    draw = ImageDraw.Draw(chart)
    for x in (width // 4, width // 2, width * 3 // 4):          # light lines at 64, 128 and 192
        draw.line([(x, 0), (x, height)], fill=BORDER)
    draw.rectangle((0, 0, width - 1, height - 1), outline=BORDER)
    peak = max(histogram)
    if peak == 0:                                               # no image yet: an empty chart
        return chart
    for channel, color in enumerate([(239, 68, 68), (34, 197, 94), (59, 130, 246)]):    # red, green, blue
        values = histogram[channel * 256:(channel + 1) * 256]
        points = [(1 + x * (width - 3) / 255, height - 2 - count / peak * (height - 8))
                  for x, count in enumerate(values)]
        layer = Image.new("RGBA", (width, height))              # a see-through layer for this color
        layer_draw = ImageDraw.Draw(layer)
        layer_draw.polygon([(1, height - 2)] + points + [(width - 2, height - 2)], fill=color + (50,))
        layer_draw.line(points, fill=color + (255,), width=2, joint="curve")
        chart = Image.alpha_composite(chart, layer)
    return chart


# ---------------------------------------------------------------
# 9) The main window (Tkinter GUI)
# ---------------------------------------------------------------
class CaptionApp:
    def __init__(self, root):
        self.root = root
        self.image = None           # the current image (PIL)
        self.image_name = ""
        self.caption = ""
        self.answer = ""
        self.prediction = ""        # the Caption Generator's prediction caption
        self.busy = False           # True while waiting for the AI
        self.waiting_box = None     # text box that shows "waiting..." messages
        self.speech = None          # the running speech process
        self.speaking = None        # "caption", "answer" or "prediction"
        self.current_tab = 0        # 0 = Image Analysis, 1 = Caption Generator, 2 = Ask a Question

        root.title("Image Analyzer")
        root.configure(bg=BG)
        root.resizable(False, False)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.logo = ImageTk.PhotoImage(make_logo(40, "white", ACCENT))
        self.icon = ImageTk.PhotoImage(make_logo(32, ACCENT, "white"))
        root.iconphoto(True, self.icon)             # the small picture in the title bar and the taskbar

        style = ttk.Style(root)
        style.theme_use("clam")                     # a ttk look whose colors can be changed
        style.configure("Vertical.TScrollbar", background="#cbd5e1", troughcolor=BOX, bordercolor=BOX,
                        lightcolor="#cbd5e1", darkcolor="#cbd5e1", arrowcolor=MUTED, gripcount=0)
        style.map("Vertical.TScrollbar", background=[("active", LIGHT)])
        style.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor=BORDER, bordercolor=BORDER,
                        lightcolor=ACCENT, darkcolor=ACCENT, thickness=6)

        # top: title bar with a color gradient
        self.header = tk.Canvas(root, height=64, bg=ACCENT, highlightthickness=0)
        self.header.pack(fill="x")
        self.header.bind("<Configure>", self.draw_header)

        # bottom: status bar with a colored dot, a message, and a moving bar while the AI works
        status = tk.Frame(root, bg=CARD, padx=16, pady=5)
        status.pack(side="bottom", fill="x")
        tk.Frame(root, bg=BORDER, height=1).pack(side="bottom", fill="x")
        self.status_dot = tk.Label(status, text="â—", font=(FONT, 10), bg=CARD, fg=LIGHT)
        self.status_dot.pack(side="left")
        self.status_label = tk.Label(status, text="", font=(FONT, 9), bg=CARD, fg=MUTED)
        self.status_label.pack(side="left", padx=(4, 0))
        tk.Label(status, text="Ctrl+O  open image      Ctrl+V  paste image", font=(FONT, 9),
                 bg=CARD, fg=LIGHT).pack(side="right")
        self.progress = ttk.Progressbar(status, mode="indeterminate", length=140)

        main = tk.Frame(root, bg=BG, padx=20, pady=16)
        main.pack(fill="both", expand=True)

        # toolbar: main buttons on the left, options on the right
        bar = make_card(main, padding=12)
        bar.pack(fill="x", pady=(0, 16))
        self.upload_btn = make_button(bar, "ðŸ“‚  Upload Image", self.upload_image)
        self.retry_btn = make_button(bar, "âŸ³  Try Again", self.start_caption, "soft")
        self.model_var = tk.StringVar(value=ONLINE)
        self.length_var = tk.StringVar(value="Short")
        self.length_choice = make_choice(bar, self.length_var, ["Short", "Detailed"], self.option_changed)
        self.length_choice.pack(side="right")
        tk.Label(bar, text="Caption", font=(FONT, 10), bg=CARD, fg=MUTED).pack(side="right", padx=(24, 8))
        self.model_choice = make_choice(bar, self.model_var, [ONLINE, OFFLINE], self.option_changed)
        self.model_choice.pack(side="right")
        tk.Label(bar, text="AI model", font=(FONT, 10), bg=CARD, fg=MUTED).pack(side="right", padx=(0, 8))

        # middle: image and caption on the left, tabs on the right
        body = tk.Frame(main, bg=BG)
        body.pack(fill="both", expand=True)
        left, right = make_card(body), make_card(body, padding=0)
        left.pack(side="left", fill="y")
        right.pack(side="left", fill="both", expand=True, padx=(16, 0))
        self.build_image_card(left)
        self.build_tabs(right)

        for key in ("<Control-o>", "<Control-O>"):          # keyboard shortcuts
            root.bind(key, lambda e: self.upload_image())
        for key in ("<Control-v>", "<Control-V>"):
            root.bind(key, self.paste_image)
        self.set_status("Ready. Upload an image to start.")
        self.update_buttons()

        root.update_idletasks()                     # put the window in the middle of the screen
        x = (root.winfo_screenwidth() - root.winfo_reqwidth()) // 2
        y = max(0, (root.winfo_screenheight() - root.winfo_reqheight()) // 2 - 30)
        root.geometry(f"+{x}+{y}")

    def draw_header(self, event):
        """Title bar: a color gradient (indigo to purple) with the logo, the title and a subtitle."""
        canvas, width, height = self.header, event.width, event.height
        canvas.delete("all")
        start, end = (79, 70, 229), (147, 51, 234)
        for x in range(0, width, 4):                # thin stripes, each one a little more purple
            t = x / width
            color = "#" + "".join(f"{round(a + (b - a) * t):02x}" for a, b in zip(start, end))
            canvas.create_rectangle(x, 0, x + 4, height, fill=color, width=0)
        canvas.create_image(20, height / 2, image=self.logo, anchor="w")
        canvas.create_text(72, height / 2 - 10, text="Image Analyzer", font=(BOLD, 15),
                           fill="white", anchor="w")
        canvas.create_text(73, height / 2 + 13, font=(FONT, 9), fill="#e0e7ff", anchor="w",
                           text="Upload an image and let AI describe it, analyze it and predict what is happening")
        canvas.create_text(width - 20, height / 2, text="DIP Project", font=(BOLD, 10), fill="#e0e7ff", anchor="e")

    def build_image_card(self, card):
        row = tk.Frame(card, bg=CARD)
        row.pack(fill="x", pady=(0, 8))
        tk.Label(row, text="Image", font=(BOLD, 11), bg=CARD, fg=TEXT).pack(side="left")
        self.file_label = tk.Label(row, text="", font=(FONT, 9), bg=CARD, fg=MUTED)
        self.file_label.pack(side="right")

        # the picture area; before an image is chosen it shows a dashed box that says where to click
        self.preview = tk.Canvas(card, width=PREVIEW_W, height=PREVIEW_H, bg=BOX, highlightthickness=0,
                                 cursor="hand2")
        self.preview.pack()
        self.preview.bind("<Button-1>", lambda e: self.upload_image())
        w, h = PREVIEW_W, PREVIEW_H
        self.preview.create_rectangle(12, 12, w - 12, h - 12, outline="#cbd5e1", dash=(5, 4))
        self.preview.create_text(w / 2, h / 2 - 30, text="ðŸ–¼", font=(FONT, 30), fill=LIGHT)
        self.preview.create_text(w / 2, h / 2 + 18, text="Click here to upload an image", font=(BOLD, 12),
                                 fill=TEXT)
        self.preview.create_text(w / 2, h / 2 + 42, font=(FONT, 9), fill=MUTED,
                                 text="or press Ctrl+O,  or paste a copied picture with Ctrl+V")

        row = tk.Frame(card, bg=CARD)
        row.pack(fill="x", pady=(16, 8))
        tk.Label(row, text="Caption", font=(BOLD, 11), bg=CARD, fg=TEXT).pack(side="left")
        self.source_label = tk.Label(row, text="", font=(BOLD, 8), bg=CARD, fg=ACCENT_DARK, padx=8, pady=1)
        self.source_label.pack(side="right")        # which AI made the caption, and how fast
        frame, self.caption_box = make_text_box(card, PREVIEW_W, 130, 13)
        frame.pack()
        self.set_text(self.caption_box, "Your caption will appear here.", LIGHT)
        row = tk.Frame(card, bg=CARD)
        row.pack(fill="x", pady=(12, 0))
        self.copy_btn = make_button(row, "â§‰  Copy", lambda: self.copy_text(self.caption), "soft")
        self.speak_btn = make_button(row, "ðŸ”Š  Speak", lambda: self.toggle_speech("caption"), "soft", width=9)
        self.save_btn = make_button(row, "ðŸ’¾  Save Image", lambda: self.save_image(self.caption, "captioned"), "soft")

    def build_tabs(self, card):
        """Tabs on the right: click a name to show its page (the chosen one is underlined)."""
        bar = tk.Frame(card, bg=CARD, padx=10)
        bar.pack(fill="x")
        tk.Frame(card, bg=BORDER, height=1).pack(fill="x")         # line under the tab names
        holder = tk.Frame(card, bg=CARD)
        holder.pack(fill="both", expand=True)
        holder.grid_rowconfigure(0, weight=1)
        holder.grid_columnconfigure(0, weight=1)
        self.tab_names, self.pages = [], []
        for i, name in enumerate(["ðŸ“Š  Image Analysis", "ðŸ’­  Caption Generator", "ðŸ’¬  Ask a Question"]):
            tab = tk.Frame(bar, bg=CARD)
            tab.pack(side="left")
            label = tk.Label(tab, text=name, font=(BOLD, 10), bg=CARD, fg=MUTED, padx=8, pady=10, cursor="hand2")
            label.pack()
            label.line = tk.Frame(tab, bg=CARD, height=3)          # colored line under the chosen tab
            label.line.pack(fill="x")
            label.bind("<Button-1>", lambda e, i=i: self.select_tab(i))
            label.bind("<Enter>", lambda e, i=i: self.hover_tab(i, True))
            label.bind("<Leave>", lambda e, i=i: self.hover_tab(i, False))
            page = tk.Frame(holder, bg=CARD, padx=18, pady=16)
            page.grid(row=0, column=0, sticky="nsew")              # all pages in the same place
            self.tab_names.append(label)
            self.pages.append(page)
        self.build_analysis_tab(self.pages[0])
        self.build_prediction_tab(self.pages[1])
        self.build_ask_tab(self.pages[2])
        self.select_tab(0)

    def select_tab(self, index):
        self.current_tab = index
        for i, label in enumerate(self.tab_names):
            label.configure(fg=ACCENT_DARK if i == index else MUTED)
            label.line.configure(bg=ACCENT if i == index else CARD)
        self.pages[index].tkraise()                 # bring this page to the front
        (self.question_entry if index == 2 else self.root).focus_set()

    def hover_tab(self, index, inside):
        if index != self.current_tab:
            self.tab_names[index].configure(fg=TEXT if inside else MUTED)

    def build_analysis_tab(self, tab):
        tab.grid_columnconfigure(1, weight=1)       # names on the left, values on the right
        self.info, self.meters = {}, {}
        for row, name in [(0, "Size"), (2, "Brightness"), (4, "Contrast")]:
            top = 0 if row == 0 else 10
            tk.Label(tab, text=name, font=(BOLD, 10), bg=CARD, fg=TEXT).grid(row=row, column=0, sticky="w",
                                                                             pady=(top, 0))
            self.info[name] = tk.Label(tab, text="-", font=(FONT, 10), bg=CARD, fg=MUTED)
            self.info[name].grid(row=row, column=1, sticky="e", pady=(top, 0))
            if name != "Size":                      # a bar that shows the value
                self.meters[name] = tk.Canvas(tab, width=TAB_W, height=10, bg=CARD, highlightthickness=0)
                self.meters[name].grid(row=row + 1, column=0, columnspan=2, pady=(6, 0))
                draw_meter(self.meters[name], 0)

        tk.Label(tab, text="Main colors", font=(BOLD, 10), bg=CARD, fg=TEXT).grid(
            row=6, column=0, columnspan=2, sticky="w", pady=(18, 6))
        tiles = tk.Frame(tab, bg=CARD)
        tiles.grid(row=7, column=0, columnspan=2, sticky="w")
        self.color_tiles = []
        for i in range(5):
            tile = tk.Frame(tiles, bg=CARD)
            tile.pack(side="left", padx=(0 if i == 0 else 10, 0))
            swatch = tk.Frame(tile, width=(TAB_W - 40) // 5, height=34, bg=BOX,
                              highlightthickness=1, highlightbackground=BORDER)
            swatch.pack()
            text = tk.Label(tile, text=" \n ", font=("Consolas", 8), bg=CARD, fg=MUTED)
            text.pack(pady=(3, 0))
            self.color_tiles.append((swatch, text))

        tk.Label(tab, text="RGB histogram", font=(BOLD, 10), bg=CARD, fg=TEXT).grid(
            row=8, column=0, columnspan=2, sticky="w", pady=(14, 6))
        self.histogram = tk.Canvas(tab, width=TAB_W, height=HIST_H, bg=CARD, highlightthickness=0)
        self.histogram.grid(row=9, column=0, columnspan=2)
        self.histogram_photo = ImageTk.PhotoImage(draw_histogram([0] * 768, TAB_W, HIST_H))
        self.histogram.create_image(0, 0, image=self.histogram_photo, anchor="nw")
        self.histogram.create_text(TAB_W / 2, HIST_H / 2, text="No image yet", font=(FONT, 9), fill=LIGHT)
        axis = tk.Frame(tab, bg=CARD)
        axis.grid(row=10, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        tk.Label(axis, text="0  dark", font=(FONT, 8), bg=CARD, fg=LIGHT).pack(side="left")
        tk.Label(axis, text="bright  255", font=(FONT, 8), bg=CARD, fg=LIGHT).pack(side="right")

    def build_prediction_tab(self, tab):
        tk.Label(tab, text="Prediction caption", font=(BOLD, 11), bg=CARD, fg=TEXT).pack(anchor="w")
        tk.Label(tab, text="Guesses what the person is thinking, feeling or about to do, "
                           "from clues like the face, pose, hands, objects and place.",
                 font=(FONT, 9), bg=CARD, fg=MUTED, justify="left", wraplength=TAB_W).pack(anchor="w", pady=(2, 12))
        row = tk.Frame(tab, bg=CARD)
        row.pack(fill="x")
        self.predict_btn = make_button(row, "âœ¨  Generate Prediction", self.start_prediction)
        frame, self.prediction_box = make_text_box(tab, TAB_W, 250, 12)
        frame.pack(pady=(12, 0))
        self.set_text(self.prediction_box, "Upload an image, then click 'Generate Prediction'.", LIGHT)
        row = tk.Frame(tab, bg=CARD)
        row.pack(fill="x", pady=(12, 0))
        self.copy_prediction_btn = make_button(row, "â§‰  Copy", lambda: self.copy_text(self.prediction), "soft")
        self.speak_prediction_btn = make_button(row, "ðŸ”Š  Speak", lambda: self.toggle_speech("prediction"),
                                                "soft", width=9)
        self.save_prediction_btn = make_button(row, "ðŸ’¾  Save Image",
                                               lambda: self.save_image(self.prediction, "prediction"), "soft")
        tk.Label(tab, text="â“˜  Always uses the online model (Qwen 27B).", font=(FONT, 8),
                 bg=CARD, fg=LIGHT).pack(anchor="w", pady=(12, 0))

    def build_ask_tab(self, tab):
        tk.Label(tab, text="Ask anything about the image", font=(BOLD, 11), bg=CARD, fg=TEXT).pack(anchor="w")
        row = tk.Frame(tab, bg=CARD)
        row.pack(fill="x", pady=(10, 0))
        self.question_entry = tk.Entry(row, font=(FONT, 11), relief="flat", bd=7, bg=BOX, fg=TEXT,
                                       insertbackground=TEXT, highlightthickness=1,
                                       highlightbackground=BORDER, highlightcolor=ACCENT)
        self.question_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.question_entry.bind("<Return>", self.ask)
        self.ask_btn = make_button(row, "âž¤  Ask", self.ask)
        self.ask_btn.pack_configure(padx=0)
        row = tk.Frame(tab, bg=CARD)
        row.pack(fill="x", pady=(10, 0))
        tk.Label(row, text="Try:", font=(FONT, 9), bg=CARD, fg=MUTED).pack(side="left", padx=(0, 6))
        self.example_btns = [make_button(row, question, lambda q=question: self.ask_example(q), "soft", small=True)
                             for question in ("How many people?", "What is happening?", "Where is this?")]
        frame, self.answer_box = make_text_box(tab, TAB_W, 250, 12)
        frame.pack(pady=(12, 0))
        self.set_text(self.answer_box, "Upload an image, then ask a question about it.", LIGHT)
        row = tk.Frame(tab, bg=CARD)
        row.pack(fill="x", pady=(12, 0))
        self.speak_answer_btn = make_button(row, "ðŸ”Š  Speak Answer", lambda: self.toggle_speech("answer"),
                                            "soft", width=15)
        tk.Label(tab, text="â“˜  Questions always use the online model (Qwen 27B).", font=(FONT, 8),
                 bg=CARD, fg=LIGHT).pack(anchor="w", pady=(12, 0))

    # ----- small helpers -----
    def update_buttons(self):
        """Enable or disable buttons depending on what is happening (disabled buttons turn pale)."""
        idle, has_image = not self.busy, self.image is not None
        set_enabled(self.upload_btn, idle)
        for button in [self.retry_btn, self.ask_btn, self.predict_btn] + self.example_btns:
            set_enabled(button, idle and has_image)
        for button in (self.copy_btn, self.speak_btn, self.save_btn):
            set_enabled(button, bool(self.caption))
        for button in (self.copy_prediction_btn, self.speak_prediction_btn, self.save_prediction_btn):
            set_enabled(button, bool(self.prediction))
        set_enabled(self.speak_answer_btn, bool(self.answer))
        set_choice_enabled(self.model_choice, idle)
        set_choice_enabled(self.length_choice, idle and self.model_var.get() != OFFLINE)   # BLIP: short only
        if self.busy:                               # moving bar in the status bar while the AI works
            self.progress.pack(side="right", padx=(0, 16))
            self.progress.start(15)
        else:
            self.progress.stop()
            self.progress.pack_forget()

    def set_status(self, text, color=LIGHT):
        """The line at the bottom: a dot (gray, amber = working, green = done, red = error) and a message."""
        self.status_dot.configure(fg=color)
        self.status_label.configure(text=text)

    def set_text(self, box, text, color=TEXT):
        """Show text in a read-only text box (short text centered, long text left-aligned)."""
        align = "center" if len(text) < 100 else "left"
        box.configure(state="normal")
        box.delete("1.0", "end")
        box.tag_configure("style", foreground=color, justify=align)
        box.insert("1.0", text, "style")
        box.configure(state="disabled")

    def run_in_background(self, work, on_done):
        """Run slow work (AI requests) in another thread, so the window does not freeze."""
        def runner():
            start = time.time()
            try:
                result, error = work(), None
            except Exception as e:
                result, error = None, str(e)
            try:
                self.root.after(0, on_done, result, error, time.time() - start)
            except (RuntimeError, tk.TclError):
                pass                                # window was closed while waiting
        threading.Thread(target=runner, daemon=True).start()

    def show_waiting(self, seconds):
        """Called from the background thread when the free limit is reached and the program waits."""
        text = f"Free limit reached, waiting {seconds:.0f} seconds..."

        def show():
            self.set_text(self.waiting_box, text, MUTED)
            self.set_status(text, AMBER)
        self.root.after(0, show)

    # ----- getting an image -----
    def upload_image(self):
        if self.busy:
            return
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.webp *.tif *.tiff"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            image = load_image(path)
        except Exception:
            messagebox.showerror("Error", "This file is not a supported image.")
            return
        self.set_image(image, os.path.basename(path))

    def paste_image(self, event=None):
        """Ctrl+V: use a picture from the clipboard (a screenshot, or an image file copied in File Explorer)."""
        if self.busy:
            return
        try:
            clip = ImageGrab.grabclipboard()
        except Exception:
            return
        if isinstance(clip, Image.Image):
            self.set_image(to_rgb(clip), "pasted_image.png")
        elif isinstance(clip, list):                # copied files: open the first one that is an image
            for path in clip:
                try:
                    image = load_image(path)
                except Exception:
                    continue
                self.set_image(image, os.path.basename(path))
                return

    def set_image(self, image, name):
        self.stop_speech()
        self.image, self.image_name = image, name
        preview = image.copy()
        preview.thumbnail((PREVIEW_W, PREVIEW_H))
        self.photo = ImageTk.PhotoImage(preview)    # keep reference so it isn't garbage-collected
        self.preview.delete("all")
        self.preview.create_image(PREVIEW_W / 2, PREVIEW_H / 2, image=self.photo)
        short_name = name if len(name) <= 40 else name[:37] + "..."
        self.file_label.configure(text=f"{short_name}  Â·  {image.width} Ã— {image.height}")
        self.show_analysis(analyze_image(image))
        self.answer, self.prediction = "", ""
        self.set_text(self.answer_box, "Ask a question about this picture.", LIGHT)
        self.set_text(self.prediction_box, "Click 'Generate Prediction' to guess what is going on in this picture.",
                      LIGHT)
        self.start_caption()

    # ----- caption -----
    def option_changed(self):
        if self.model_var.get() == OFFLINE:
            self.length_var.set("Short")            # BLIP only makes short captions
            threading.Thread(target=self.preload_blip, daemon=True).start()   # start loading it now
        self.update_buttons()
        if self.image is not None and not self.busy:
            self.start_caption()                    # make a new caption with the chosen options

    def start_caption(self):
        if self.speaking == "caption":
            self.stop_speech()
        self.caption, self.busy = "", True
        self.update_buttons()
        self.waiting_box = self.caption_box
        self.source_label.configure(text="", bg=CARD)
        model = self.model_var.get()
        if model == OFFLINE and blip is None:
            message = "Loading the BLIP model (only the first time, about 40 seconds)..."
        else:
            message = "Generating caption..."
        self.set_text(self.caption_box, message, MUTED)
        self.set_status(message, AMBER)
        image, length = self.image, self.length_var.get()
        self.run_in_background(lambda: generate_caption(image, length, model), self.show_caption)

    def preload_blip(self):
        try:
            get_blip()
        except Exception:
            pass                                    # any problem is shown later, when a caption is made

    def show_caption(self, result, error, seconds):
        self.busy = False
        if error:
            self.set_text(self.caption_box, "Could not generate caption. Press 'Try Again'.", RED)
            self.set_status("Could not generate caption.", RED)
        else:
            self.caption, source = result
            self.set_text(self.caption_box, self.caption)
            self.source_label.configure(text=f"{source}  Â·  {seconds:.1f} s", bg=ACCENT_SOFT)
            self.set_status(f"Caption ready  Â·  made by {source} in {seconds:.1f} s", GREEN)
        self.update_buttons()
        if error:
            if self.model_var.get() == ONLINE:
                error += "\n\nTip: choose AI model 'BLIP (offline)' to make captions without internet or limits."
            messagebox.showerror("Could not generate caption", error)

    def copy_text(self, text):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.set_status("Copied to the clipboard.", GREEN)

    def save_image(self, text, suffix):
        """Save the image with the given caption (or prediction) written under it."""
        name = os.path.splitext(self.image_name)[0]
        path = filedialog.asksaveasfilename(
            title="Save image with caption", initialfile=f"{name}_{suffix}.jpg", defaultextension=".jpg",
            filetypes=[("JPEG image", "*.jpg"), ("PNG image", "*.png")],
        )
        if not path:
            return
        try:
            add_caption_to_image(self.image, text).save(path)
        except Exception as e:
            messagebox.showerror("Could not save", str(e))
            return
        self.set_status(f"Saved: {os.path.basename(path)}", GREEN)

    # ----- questions -----
    def ask(self, event=None):
        question = self.question_entry.get().strip()
        if not question or self.image is None or self.busy:
            return
        if self.speaking == "answer":
            self.stop_speech()
        self.answer, self.busy = "", True
        self.update_buttons()
        self.waiting_box = self.answer_box
        self.set_text(self.answer_box, "Thinking...", MUTED)
        self.set_status("Thinking about your question...", AMBER)
        image = self.image
        self.run_in_background(lambda: ask_question(image, question), self.show_answer)

    def ask_example(self, question):
        """An example question was clicked: put it in the box and ask it."""
        self.question_entry.delete(0, "end")
        self.question_entry.insert(0, question)
        self.ask()

    def show_answer(self, result, error, seconds):
        self.busy = False
        if error:
            self.set_text(self.answer_box, "Could not get an answer. Try again.", RED)
            self.set_status("Could not get an answer.", RED)
        else:
            self.answer, source = result
            self.set_text(self.answer_box, self.answer)
            self.set_status(f"Answer ready  Â·  made by {source} in {seconds:.1f} s", GREEN)
        self.update_buttons()
        if error:
            messagebox.showerror("Could not get an answer", error)

    # ----- Caption Generator (prediction caption) -----
    def start_prediction(self):
        if self.image is None or self.busy:
            return
        if self.speaking == "prediction":
            self.stop_speech()
        self.prediction, self.busy = "", True
        self.update_buttons()
        self.waiting_box = self.prediction_box
        self.set_text(self.prediction_box, "Predicting...", MUTED)
        self.set_status("Predicting what is going on...", AMBER)
        image = self.image
        self.run_in_background(lambda: predict_caption(image), self.show_prediction)

    def show_prediction(self, result, error, seconds):
        self.busy = False
        if error:
            self.set_text(self.prediction_box, "Could not make a prediction. Try again.", RED)
            self.set_status("Could not make a prediction.", RED)
        else:
            self.prediction, source = result
            self.set_text(self.prediction_box, self.prediction)
            self.set_status(f"Prediction ready  Â·  made by {source} in {seconds:.1f} s", GREEN)
        self.update_buttons()
        if error:
            messagebox.showerror("Could not make a prediction", error)

    # ----- image analysis -----
    def show_analysis(self, info):
        width, height = info["size"]
        self.info["Size"].configure(text=f"{width} Ã— {height} pixels")
        for name, key in (("Brightness", "brightness"), ("Contrast", "contrast")):
            self.info[name].configure(text=f"{info[key]:.0f}%  Â·  {info[key + '_word']}")
            draw_meter(self.meters[name], info[key])

        for i, (swatch, text) in enumerate(self.color_tiles):
            if i < len(info["colors"]):
                (r, g, b), percent = info["colors"][i]
                hex_color = f"#{r:02X}{g:02X}{b:02X}"
                swatch.configure(bg=hex_color)
                text.configure(text=f"{hex_color}\n{percent:.1f}%")
            else:
                swatch.configure(bg=BOX)
                text.configure(text=" \n ")

        # histogram: for every brightness value 0-255, how many pixels have it (one curve per color)
        self.histogram_photo = ImageTk.PhotoImage(draw_histogram(info["histogram"], TAB_W, HIST_H))
        self.histogram.delete("all")
        self.histogram.create_image(0, 0, image=self.histogram_photo, anchor="nw")

    # ----- reading aloud -----
    def toggle_speech(self, which):
        """Read the caption, answer or prediction aloud; press again to stop."""
        if self.speaking == which:
            self.stop_speech()
            return
        self.stop_speech()
        text = {"caption": self.caption, "answer": self.answer, "prediction": self.prediction}[which]
        button = {"caption": self.speak_btn, "answer": self.speak_answer_btn,
                  "prediction": self.speak_prediction_btn}[which]
        try:
            self.speech = speak(text)
        except Exception as e:
            messagebox.showinfo("Speak", str(e))
            return
        self.speaking = which
        button.configure(text="â–   Stop")
        self.check_speech(self.speech)

    def check_speech(self, process):
        if process is not self.speech:
            return                                  # a newer speech has started
        if process.poll() is None:
            self.root.after(300, self.check_speech, process)
        else:
            self.stop_speech()                      # finished speaking

    def stop_speech(self):
        if self.speech and self.speech.poll() is None:
            self.speech.terminate()
        self.speech, self.speaking = None, None
        for button in (self.speak_btn, self.speak_answer_btn, self.speak_prediction_btn):
            button.configure(text=button.label)     # "Stop" -> back to "Speak" / "Speak Answer"

    def close(self):
        self.stop_speech()
        self.root.destroy()


# ---------------------------------------------------------------
# 10) Start the program
# ---------------------------------------------------------------
if __name__ == "__main__":
    load_env()
    if len(sys.argv) > 1:                           # terminal mode: caption every image given
        on_wait = lambda seconds: print(f"Free limit reached, waiting {seconds:.0f} seconds...")
        model = OFFLINE if "--offline" in sys.argv else ONLINE
        for path in [arg for arg in sys.argv[1:] if not arg.startswith("--")]:
            try:
                if "--predict" in sys.argv:             # Caption Generator (always online)
                    caption, source = predict_caption(load_image(path))
                else:
                    caption, source = generate_caption(load_image(path), model=model)
                print(f"{os.path.basename(path)}: {caption}")
            except Exception as e:
                print(f"{os.path.basename(path)}: ERROR - {e}")
    else:
        root = tk.Tk()
        app = CaptionApp(root)
        on_wait = app.show_waiting
        root.mainloop()

