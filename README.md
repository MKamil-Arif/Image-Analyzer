# Image Analyzer

A Python desktop app for image analysis, AI captions, and questions about an image.

## Features

- View image size, brightness, contrast, main colors, and RGB histogram.
- Generate captions with an online AI model or local BLIP model.
- Ask questions about an image and save or copy results.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Add your Groq or Gemini API key to `.env` for online features.

## Run

```powershell
python image_caption_generator.py
```

For local captions, install the optional packages with `python -m pip install torch transformers`. The BLIP model downloads the first time it is used.

Keep API keys in `.env`; do not upload them to GitHub.
