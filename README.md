# SD Nano - Local Kandinsky 2.2 Auto-Mask Inpainting Web App

## Project Structure

- `backend/main.py` - FastAPI service that runs CLIPSeg auto-segmentation + Kandinsky 2.2 inpainting
- `backend/requirements.txt` - Python dependencies
- `frontend/index.html` - Web UI
- `frontend/app.js` - Upload, prompt/target controls, and API call logic

## Run Backend

1. Create/activate a Python virtualenv (recommended).
2. Install dependencies:

```bash
cd backend
pip install -r requirements.txt
```

3. Start API server:

```bash
uvicorn main:app --reload
```

On first run, model files are downloaded from Hugging Face.

## Run Frontend

Open this file in your browser:

- `frontend/index.html`

## Usage

1. Upload an image.
2. Enter a mask target (e.g. `fingernails`, `shirt`, `hair`).
3. Enter an edit prompt (e.g. `glossy silver nails`).
4. Adjust threshold if needed (lower = broader mask, higher = tighter mask).
5. Click **Generate Edit**.
6. The app shows original image, auto-generated mask, and final result.

## Notes

- Backend URL is hardcoded in frontend as `http://127.0.0.1:8000/generate-edit/`.
- Apple Silicon uses MPS automatically when available.
- API expects form fields: `image`, `target`, `prompt`, `threshold`.
