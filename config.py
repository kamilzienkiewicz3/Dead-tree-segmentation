import numpy as np
import os

# --- ŚCIEŻKI DO PLIKÓW ---
BASE_DATA_DIR = "data"
SEGMENTATION_DIR = os.path.join(BASE_DATA_DIR, "USA_segmentation")

PATHS = {
    "NRG": os.path.join(SEGMENTATION_DIR, "NRG_images", "*.png"),
    "MASKS": os.path.join(SEGMENTATION_DIR, "masks", "*.png"),
    "RGB": os.path.join(SEGMENTATION_DIR, "RGB_images", "*.png"),
    "OUTPUT": "./generated_masks"
}

# --- PARAMETRY DETEKCJI NRG (Percentyle) ---
# Wartość 35 oznacza 35-ty percentyl (z Twojego kodu w Colab)
NDVI_PERCENTILE = 35 

# --- PARAMETRY DETEKCJI RGB (Las - HSV) ---
# Kolory lasu zdefiniowane w Twoim kodzie
FOREST_LOWER_PURPLE = np.array([0.65, 0.15, 0.25])
FOREST_UPPER_PURPLE = np.array([0.90, 1.00, 1.00])

# --- PARAMETRY CZYSZCZENIA MASEK ---
# Wielkość "pędzla" do usuwania szumów (3x3)
MORPH_KERNEL_SIZE = (3, 3)

# Minimalna wielkość obiektu (wszystko mniejsze niż 50 pikseli znika)
MIN_OBJECT_SIZE = 50

# --- USTAWIENIA EKSPERYMENTU ---
NUM_SAMPLES_TO_PROCESS = 5