"""Shared configuration for AgriSight.

CLASS_MAP: PlantVillage folder name -> PlantDoc folder name.
Only classes that exist in BOTH datasets are used, so the PlantVillage -> PlantDoc
cross-dataset test is a fair comparison. Run `python check_classes.py` first; if a
folder name differs on your copy of the dataset, just edit the right-hand side here.
"""
IMG_SIZE = 224
SEED = 42

CLASS_MAP = {
    "Tomato___Early_blight":                 "Tomato Early blight leaf",
    "Tomato___Late_blight":                  "Tomato leaf late blight",
    "Tomato___Septoria_leaf_spot":           "Tomato Septoria leaf spot",
    "Tomato___Bacterial_spot":               "Tomato leaf bacterial spot",
    "Tomato___Tomato_mosaic_virus":          "Tomato leaf mosaic virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato leaf yellow virus",
    "Tomato___healthy":                      "Tomato leaf",
    "Potato___Early_blight":                 "Potato leaf early blight",
    "Potato___Late_blight":                  "Potato leaf late blight",
    "Corn_(maize)___Northern_Leaf_Blight":   "Corn leaf blight",
    "Corn_(maize)___Common_rust_":           "Corn rust leaf",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Corn Gray leaf spot",
    "Apple___Apple_scab":                    "Apple Scab Leaf",
    "Apple___Cedar_apple_rust":              "Apple rust leaf",
    "Apple___healthy":                       "Apple leaf",
    "Grape___Black_rot":                     "grape leaf black rot",
    "Pepper,_bell___Bacterial_spot":         "Bell_pepper leaf spot",
    "Pepper,_bell___healthy":                "Bell_pepper leaf",
}

# Max images per class taken from PlantVillage (keeps laptop / Kaggle training fast)
MAX_PER_CLASS = 600
