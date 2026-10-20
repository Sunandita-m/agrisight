# AgriSight: run frontend + real Kaggle-trained model

## Important status
The supplied notebook and uploaded `agrisight_results.zip` confirmed training completed. This project includes the trained `mobilenet_field.pt` checkpoint at `backend/models/mobilenet_field.pt`. The checkpoint uses MobileNetV3 and has 18 output classes.

The notebook's evaluation reports a substantial PlantDoc field-image domain gap, so treat predictions as experimental and do not rely on them alone for pesticide or crop-treatment decisions.

## 1. Model checkpoint
The trained checkpoint is already included at `backend/models/mobilenet_field.pt`. It contains `model`, `classes`, and `state_dict` metadata. The API defaults to this path if `AGRISIGHT_MODEL_CHECKPOINT` is not set.

## 2. Start the Python API (Windows PowerShell)
From the project root:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
$env:AGRISIGHT_MODEL_CHECKPOINT = "$PWD\backend\models\mobilenet_field.pt"
$env:AGRISIGHT_ALLOWED_ORIGINS = "http://localhost:3000"
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
Check `http://localhost:8000/health`. It should report `model_configured: true` and `model_status: ready`.

## 3. Start the frontend
Open a second PowerShell in the project root:
```powershell
npm install
$env:NEXT_PUBLIC_AGRISIGHT_API_URL = "http://localhost:8000"
npm run dev
```
Open `http://localhost:3000`.

## 4. Verify a real prediction
Upload a genuine JPEG or PNG leaf photo. The UI calls `POST /predict` with multipart field `file`. I verified the health endpoint reports the model ready and a sample image returns a prediction from the real checkpoint. Grad-CAM is optional; if its package is unavailable, prediction still works.

## Notes
- Never commit model checkpoints, API keys, or `.env` secrets to a public repository.
- Use a Python version compatible with the installed PyTorch build. If `py -3.11` is unavailable, install a supported Python version and adjust the command.
- The frontend can be started without the checkpoint, but genuine inference requires the checkpoint file.

## Model quality limitation
The notebook evaluation reports PlantDoc field accuracy of 16.0% for this checkpoint, despite 97.8% PlantVillage test accuracy. This indicates a substantial domain gap. Treat predictions as experimental, show the model limitation to users, and do not use it as a sole basis for pesticide or crop-treatment decisions.
