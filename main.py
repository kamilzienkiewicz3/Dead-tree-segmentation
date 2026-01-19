import yaml
import argparse
import cv2
import numpy as np
import os
import glob
import logging
import shutil
from skimage import color
from skimage.morphology import remove_small_objects

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Detekcja martwych drzew - Tryb wsadowy")
parser.add_argument("--percentile", type=int, help="Procentowa wartość percentyla NDVI")
parser.add_argument("--samples", type=int, help="Liczba próbek do przetworzenia")
parser.add_argument("--output", type=str, help="Folder wyjściowy dla masek")
args = parser.parse_args()

try:
    with open("config.yaml", 'r') as stream:
        c = yaml.safe_load(stream)
except FileNotFoundError:
    logger.critical("Nie znaleziono pliku config.yaml! Upewnij się, że plik istnieje.")
    exit(1)

class Config:
    def __init__(self, entries, cli_args):
        self.PATHS = {
            "NRG": entries['paths']['nrg_pattern'],
            "RGB": entries['paths']['rgb_pattern'],
            "MASKS": entries['paths']['masks_pattern'],
            "OUTPUT": cli_args.output if cli_args.output else entries['paths']['output_dir']
        }
        self.NDVI_PERCENTILE = cli_args.percentile if cli_args.percentile is not None else entries['detection']['ndvi_percentile']
        self.NUM_SAMPLES_TO_PROCESS = cli_args.samples if cli_args.samples is not None else entries['experiment']['num_samples']
        self.MIN_OBJECT_SIZE = entries['detection']['min_object_size']
        self.MORPH_KERNEL_SIZE = tuple(entries['detection']['morph_kernel_size'])
        self.FOREST_LOWER_PURPLE = np.array(entries['forest_hsv']['lower'])
        self.FOREST_UPPER_PURPLE = np.array(entries['forest_hsv']['upper'])

config = Config(c, args)

def reset_output_directory(path):
    if os.path.exists(path):
        logger.info(f"Czyszczenie folderu wyjściowego: {path}")
        try:
            shutil.rmtree(path)
        except Exception as e:
            logger.error(f"Nie udało się usunąć folderu {path}: {e}")
            return

    os.makedirs(path, exist_ok=True)
    logger.info(f"Utworzono pusty folder: {path}")

def detect_dead_trees_advanced(nrg_image):
    nir   = nrg_image[:, :, 0].astype(float)
    red   = nrg_image[:, :, 1].astype(float)
    green = nrg_image[:, :, 2].astype(float)

    ndvi_denom = nir + red
    ndvi_denom[ndvi_denom == 0] = 1e-6
    ndvi = (nir - red) / ndvi_denom

    gndvi_denom = nir + green
    gndvi_denom[gndvi_denom == 0] = 1e-6
    gndvi = (nir - green) / gndvi_denom

    ndvi_th = np.percentile(ndvi, config.NDVI_PERCENTILE)
    gndvi_th = np.percentile(gndvi, config.NDVI_PERCENTILE)

    mask = (ndvi < ndvi_th) & (gndvi < gndvi_th)
    mask = mask.astype(np.uint8)
    kernel = np.ones(config.MORPH_KERNEL_SIZE, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

def generate_forest_mask(rgb_image):
    hsv_image = color.rgb2hsv(rgb_image)
    mask_hsv = np.all((hsv_image >= config.FOREST_LOWER_PURPLE) & 
                      (hsv_image <= config.FOREST_UPPER_PURPLE), axis=-1).astype(np.uint8)
    
    kernel = np.ones(config.MORPH_KERNEL_SIZE, np.uint8)
    mask_hsv_open = cv2.morphologyEx(mask_hsv, cv2.MORPH_OPEN, kernel)
    mask_hsv_close = cv2.morphologyEx(mask_hsv_open, cv2.MORPH_CLOSE, kernel)
    return remove_small_objects(mask_hsv_close.astype(bool), min_size=config.MIN_OBJECT_SIZE).astype(np.uint8)

def create_paired_files(paths_rgb_pattern, paths_nrg_pattern):
    rgb_dir = os.path.dirname(paths_rgb_pattern)
    nrg_dir = os.path.dirname(paths_nrg_pattern)
    
    rgb_files = sorted(glob.glob(paths_rgb_pattern))
    paired = []
    logger.info(f"Szukanie par dla {len(rgb_files)} plików...")

    for rgb_path in rgb_files:
        filename = os.path.basename(rgb_path)
        if filename.startswith("RGB_"):
            common_id = filename[4:] 
            nrg_path = os.path.join(nrg_dir, "NRG_" + common_id)
            
            if os.path.exists(nrg_path):
                paired.append((rgb_path, nrg_path))
    
    logger.info(f"Znaleziono {len(paired)} kompletnych par (RGB + NRG).")
    return paired

def process_and_save_masks(paired_files, num_samples, output_dir):
    limit = min(num_samples, len(paired_files))
    
    for i in range(limit):
        rgb_path, nrg_path = paired_files[i]
        fname = os.path.basename(rgb_path)
        file_id = fname.replace("RGB_", "").replace(".png", "")
        
        logger.info(f"[{i+1}/{limit}] Generowanie masek dla ID: {file_id}")

        try:
            rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            nrg = cv2.cvtColor(cv2.imread(nrg_path), cv2.COLOR_BGR2RGB)

            nrg_mask = detect_dead_trees_advanced(nrg)
            rgb_forest_mask = generate_forest_mask(rgb)
            
            combined_pre = (nrg_mask > 0) & (rgb_forest_mask > 0)
            combined_mask = remove_small_objects(combined_pre, min_size=config.MIN_OBJECT_SIZE).astype(np.uint8)

            cv2.imwrite(os.path.join(output_dir, f"NRG_mask_{file_id}.png"), nrg_mask * 255)
            cv2.imwrite(os.path.join(output_dir, f"RGB_forest_{file_id}.png"), rgb_forest_mask * 255)
            cv2.imwrite(os.path.join(output_dir, f"FINAL_combined_{file_id}.png"), combined_mask * 255)

        except Exception as e:
            logger.error(f"Błąd podczas przetwarzania pliku {fname}: {e}")

    logger.info("Przetwarzanie zakończone sukcesem.")

if __name__ == '__main__':
    reset_output_directory(config.PATHS["OUTPUT"])

    paired = create_paired_files(config.PATHS["RGB"], config.PATHS["NRG"])

    if paired:
        process_and_save_masks(paired, num_samples=config.NUM_SAMPLES_TO_PROCESS, output_dir=config.PATHS["OUTPUT"])
    else:
        logger.error("BŁĄD: Nie znaleziono par plików! Sprawdź ścieżki w config.yaml.")