import yaml
import argparse
import cv2
import numpy as np
import os
import glob
import logging
import shutil
import matplotlib.pyplot as plt
from skimage import color
from skimage.morphology import remove_small_objects


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(), logging.FileHandler('app.log', mode='w', encoding='utf-8')])
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description="Dead Tree Detection - Batch Mode")
parser.add_argument("--percentile", type=int, help="NDVI percentile value")
parser.add_argument("--samples", type=int, help="Number of samples to process")
parser.add_argument("--output", type=str, help="Output directory for masks")
args = parser.parse_args()


try:
    with open("config.yaml", 'r') as stream:
        config_data = yaml.safe_load(stream)
except FileNotFoundError:
    logger.critical("config.yaml not found!")
    exit(1)


class Config:
    def __init__(self, data, args):
        self.PATHS = {
            "NRG": data['paths']['nrg_pattern'],
            "RGB": data['paths']['rgb_pattern'],
            "MASKS": data['paths']['masks_pattern'],
            "OUTPUT": args.output if args.output else data['paths']['output_dir']
        }
        self.NDVI_PERCENTILE = args.percentile if args.percentile is not None else data['detection']['ndvi_percentile']
        self.NUM_SAMPLES = args.samples if args.samples is not None else data['experiment']['num_samples']
        self.MIN_OBJ_SIZE = data['detection']['min_object_size']
        self.KERNEL_SIZE = tuple(data['detection']['morph_kernel_size'])
        self.FOREST_LOWER = np.array(data['forest_hsv']['lower'])
        self.FOREST_UPPER = np.array(data['forest_hsv']['upper'])

cfg = Config(config_data, args)


def reset_output_dir(path):
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
            logger.info(f"Cleaned output directory: {path}")
        except Exception as e:
            logger.error(f"Failed to remove {path}: {e}")
            return
    os.makedirs(path, exist_ok=True)
    logger.info(f"Created empty directory: {path}")

def calculate_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / union if union > 0 else 0.0

def plot_iou_histogram(filenames, ious, output_dir):
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(filenames)), ious, color='purple', alpha=0.7)
    plt.title("IoU of Combined Mask vs GT Masks")
    plt.xlabel('GT Mask Name')
    plt.ylabel('IoU')
    plt.ylim(0, 1)
    plt.xticks(range(len(filenames)), filenames, rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    hist_path = os.path.join(output_dir, 'iou_histogram.png')
    plt.savefig(hist_path, dpi=300)
    plt.close()
    logger.info(f"IoU histogram saved: {hist_path}")


def get_dead_tree_mask(nrg_img):
    nir, red, green = nrg_img[:,:,0].astype(float), nrg_img[:,:,1].astype(float), nrg_img[:,:,2].astype(float)
    
    # Avoid division by zero
    ndvi = (nir - red) / (nir + red + 1e-6)
    gndvi = (nir - green) / (nir + green + 1e-6)

    ndvi_th = np.percentile(ndvi, cfg.NDVI_PERCENTILE)
    gndvi_th = np.percentile(gndvi, cfg.NDVI_PERCENTILE)

    mask = ((ndvi < ndvi_th) & (gndvi < gndvi_th)).astype(np.uint8)
    
    kernel = np.ones(cfg.KERNEL_SIZE, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

def get_forest_mask(rgb_img):
    hsv = color.rgb2hsv(rgb_img)
    mask = np.all((hsv >= cfg.FOREST_LOWER) & (hsv <= cfg.FOREST_UPPER), axis=-1).astype(np.uint8)
    
    kernel = np.ones(cfg.KERNEL_SIZE, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return remove_small_objects(mask.astype(bool), min_size=cfg.MIN_OBJ_SIZE).astype(np.uint8)

def find_image_pairs():
    rgb_files = sorted(glob.glob(cfg.PATHS["RGB"]))
    nrg_dir = os.path.dirname(cfg.PATHS["NRG"])
    mask_dir = os.path.dirname(cfg.PATHS["MASKS"]) if cfg.PATHS["MASKS"] else None
    
    pairs = []
    logger.info(f"Searching pairs for {len(rgb_files)} files...")

    for rgb_path in rgb_files:
        common_id = os.path.basename(rgb_path).replace("RGB_", "")
        nrg_path = os.path.join(nrg_dir, "NRG_" + common_id)
        
        if os.path.exists(nrg_path):
            mask_path = os.path.join(mask_dir, "mask_" + common_id) if mask_dir else None
            if mask_path and not os.path.exists(mask_path):
                mask_path = None
            pairs.append((rgb_path, nrg_path, mask_path))

    logger.info(f"Found {len(pairs)} pairs.")
    return pairs

def process_images(pairs):
    limit = min(cfg.NUM_SAMPLES, len(pairs))
    ious, filenames = [], []
    
    reset_output_dir(cfg.PATHS["OUTPUT"])

    for i in range(limit):
        rgb_path, nrg_path, mask_path = pairs[i]
        file_id = os.path.basename(rgb_path).replace("RGB_", "").replace(".png", "")
        
        logger.info(f"[{i+1}/{limit}] Processing ID: {file_id}")

        try:
            rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            nrg = cv2.cvtColor(cv2.imread(nrg_path), cv2.COLOR_BGR2RGB)

            dead_mask = get_dead_tree_mask(nrg)
            forest_mask = get_forest_mask(rgb)
            
            combined = remove_small_objects((dead_mask & forest_mask).astype(bool), min_size=cfg.MIN_OBJ_SIZE).astype(np.uint8)

            cv2.imwrite(os.path.join(cfg.PATHS["OUTPUT"], f"NRG_mask_{file_id}.png"), dead_mask * 255)
            cv2.imwrite(os.path.join(cfg.PATHS["OUTPUT"], f"RGB_forest_{file_id}.png"), forest_mask * 255)
            cv2.imwrite(os.path.join(cfg.PATHS["OUTPUT"], f"FINAL_combined_{file_id}.png"), combined * 255)

            if mask_path:
                gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if gt_mask is not None:
                    iou = calculate_iou(combined, gt_mask)
                    ious.append(iou)
                    filenames.append(file_id)
                    logger.info(f"IoU: {iou:.4f}")

        except Exception as e:
            logger.error(f"Error processing {file_id}: {e}")

    if ious:
        plot_iou_histogram(filenames, ious, cfg.PATHS["OUTPUT"])
        logger.info(f"Mean IoU: {np.mean(ious):.4f}")
    else:
        logger.info("No GT masks for IoU calculation.")


if __name__ == '__main__':
    pairs = find_image_pairs()
    if pairs:
        process_images(pairs)
    else:
        logger.error("No image pairs found! Check config.yaml paths.")