# AgriSight Python inference API

The Next.js frontend sends the selected JPG/PNG image to `POST /predict` as multipart field `file`. The API uses the project's original PyTorch architecture and preprocessing. It never returns a made-up prediction: without a real trained checkpoint, `/predict` returns HTTP 503.

## 1. Obtain the trained checkpoint first

The source repository does **not** include model weights. Export/download a real `.pt` file from the team's Kaggle training notebook (see `docs/backend-integration.md`). The file must be saved in the format used by `train.py`:

```python
torch.save({
    "model": model_name,
    "classes": classes,
    "state_dict": model.state_dict(),
    "aug": augmentation_name,
    "val_acc": validation_accuracy,
}, "mobilenet_field.pt")
```

Do not use `create_demo_checkpoint.py` as a production model; a randomly initialized checkpoint does not provide meaningful crop diagnosis.

## 2. Install and run (Windows PowerShell)

Run from the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
$env:AGRISIGHT_MODEL_CHECKPOINT = (Resolve-Path ".\models\mobilenet_field.pt").Path
$env:AGRISIGHT_ALLOWED_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Choose a Python version supported by the PyTorch wheels installed on your computer. If PyTorch installation fails, install the CPU or CUDA build from the official PyTorch installation selector, then install the remaining requirements.

## 3. Verify the API

Open `http://127.0.0.1:8000/health`. `model_configured` must be `true`. Open `http://127.0.0.1:8000/docs` to inspect the API. Test `/predict` with a real leaf image only after the checkpoint is configured.

## 4. Connect the frontend

Create `.env.local` in the project root:

```env
NEXT_PUBLIC_AGRISIGHT_API_URL=http://localhost:8000
```

Then restart the Next.js development server. For a deployed site, set `NEXT_PUBLIC_AGRISIGHT_API_URL` to the public HTTPS URL of the deployed API **before building the frontend**, and set `AGRISIGHT_ALLOWED_ORIGINS` to the exact frontend origin in the backend deployment environment. A `localhost` API URL only works on the same user's computer and must not be used for a public deployment.

## Important limitation

The frontend/backend source can be connected, but genuine inference cannot work until the trained checkpoint is exported from Kaggle and placed at `AGRISIGHT_MODEL_CHECKPOINT`. A Kaggle notebook URL alone does not make its in-memory model weights available to the deployed API.
