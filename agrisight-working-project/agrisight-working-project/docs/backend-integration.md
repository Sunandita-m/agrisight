# AgriSight frontend ↔ backend integration

## Current contract

- `GET /health` returns whether a checkpoint file is configured.
- `POST /predict` accepts multipart field `file` with a JPEG or PNG.
- The frontend reads `NEXT_PUBLIC_AGRISIGHT_API_URL`; local development defaults to `http://localhost:8000`.
- The backend uses `backend/agrisight/models.py` and `backend/agrisight/data.py` from the original project. Model preprocessing is resized to 224×224 and ImageNet-normalized.

## Required model artifact

The source archives provided for this integration do not contain a `.pt`, `.pth`, or `.ckpt` model artifact. The API intentionally responds with HTTP 503 until the team's trained checkpoint is exported from Kaggle and configured. The expected checkpoint dictionary has `model`, `classes`, and `state_dict` keys; the training script also saves `aug` and `val_acc`.

In the Kaggle notebook, after loading/training the final selected model, save a checkpoint in the expected format and use the notebook's file output/download mechanism to download it:

```python
import torch
checkpoint_path = "/kaggle/working/agrisight_model.pt"
torch.save({
    "model": model_name,
    "classes": classes,
    "state_dict": model.state_dict(),
    "aug": augmentation_name,
    "val_acc": validation_accuracy,
}, checkpoint_path)
print(checkpoint_path)
```

Use the actual variable names and the actual trained model from the notebook; do not paste this blindly if the notebook uses different names. Download the generated file and set `AGRISIGHT_MODEL_CHECKPOINT` to its path on the machine/server running FastAPI.

## Local development

1. Install `backend/requirements.txt` in a supported Python environment.
2. Set `AGRISIGHT_MODEL_CHECKPOINT` to the downloaded trained `.pt` file.
3. Start API: `python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000` from project root.
4. Create frontend `.env.local` with `NEXT_PUBLIC_AGRISIGHT_API_URL=http://localhost:8000`.
5. Restart Next.js, open `/health`, then test a genuine image upload.

## Deployment

The frontend and Python API must both be deployed. Set the frontend URL environment variable before the frontend build. Set backend `AGRISIGHT_ALLOWED_ORIGINS` to the frontend's exact HTTPS origin and `AGRISIGHT_MODEL_CHECKPOINT` to a checkpoint file present in the backend runtime. Do not set the frontend API URL to `localhost` for public users.
