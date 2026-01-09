import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from skimage import color
from skimage.morphology import remove_small_objects
import config
import yaml

# --- WCZYTYWANIE KONFIGURACJI YAML ---
with open("config.yaml", 'r') as stream:
    c = yaml.safe_load(stream)

# Obiekt config, który udaje stary plik .py, żeby nie zmieniać reszty kodu
class Config:
    PATHS = {
        "NRG": c['paths']['nrg_pattern'],
        "RGB": c['paths']['rgb_pattern'],
        "MASKS": c['paths']['masks_pattern'],
        "OUTPUT": c['paths']['output_dir']
    }
    NDVI_PERCENTILE = c['detection']['ndvi_percentile']
    MIN_OBJECT_SIZE = c['detection']['min_object_size']
    MORPH_KERNEL_SIZE = tuple(c['detection']['morph_kernel_size'])
    FOREST_LOWER_PURPLE = np.array(c['forest_hsv']['lower'])
    FOREST_UPPER_PURPLE = np.array(c['forest_hsv']['upper'])
    NUM_SAMPLES_TO_PROCESS = c['experiment']['num_samples']

config = Config()
# --- KONIEC SEKCJI KONFIGURACJI ---

# --- 1. FUNKCJE Z COLABA ---

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
    mask_close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask_close

def generate_forest_mask(rgb_image):
    hsv_image = color.rgb2hsv(rgb_image)
    
    mask_hsv = np.all((hsv_image >= config.FOREST_LOWER_PURPLE) & 
                      (hsv_image <= config.FOREST_UPPER_PURPLE), axis=-1).astype(np.uint8)
    
    kernel = np.ones(config.MORPH_KERNEL_SIZE, np.uint8)
    mask_hsv_open = cv2.morphologyEx(mask_hsv, cv2.MORPH_OPEN, kernel)
    mask_hsv_close = cv2.morphologyEx(mask_hsv_open, cv2.MORPH_CLOSE, kernel)

    cleaned_mask = remove_small_objects(mask_hsv_close.astype(bool), min_size=config.MIN_OBJECT_SIZE).astype(np.uint8)

    return cleaned_mask

# --- 2. PAROWANIE PLIKÓW ---
def create_paired_files(paths_rgb_pattern, paths_nrg_pattern, paths_masks_pattern):
    rgb_dir = os.path.dirname(paths_rgb_pattern)
    nrg_dir = os.path.dirname(paths_nrg_pattern)
    mask_dir = os.path.dirname(paths_masks_pattern)
    
    rgb_files = sorted(glob.glob(paths_rgb_pattern))
    paired = []
    print(f"Szukam par dla {len(rgb_files)} plików...")

    for rgb_path in rgb_files:
        filename = os.path.basename(rgb_path)
        if filename.startswith("RGB_"):
            common_id = filename[4:] 
            nrg_path = os.path.join(nrg_dir, "NRG_" + common_id)
            mask_path = os.path.join(mask_dir, "mask_" + common_id)
            if os.path.exists(nrg_path) and os.path.exists(mask_path):
                paired.append((rgb_path, nrg_path, mask_path))
    
    print(f"Znaleziono {len(paired)} kompletnych par.")
    return paired

# --- 3. METRYKI ---
def calculate_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / union if union > 0 else 0.0

def calculate_confusion_matrix_components(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    tp = np.sum(pred & gt)
    tn = np.sum(~pred & ~gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)
    return tp, tn, fp, fn

def compare_and_score_masks(paired_files, num_samples, output_dir=None):
    results = []
    if output_dir: os.makedirs(output_dir, exist_ok=True)
    
    limit = min(num_samples, len(paired_files))
    
    for i in range(limit):
        rgb_path, nrg_path, mask_path = paired_files[i]
        print(f"[{i+1}/{limit}] Analiza: {os.path.basename(rgb_path)}")

        # 1. Wczytanie BGR
        rgb_raw = cv2.imread(rgb_path)
        nrg_raw = cv2.imread(nrg_path)
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 2. Konwersja na RGB
        rgb = cv2.cvtColor(rgb_raw, cv2.COLOR_BGR2RGB)
        nrg = cv2.cvtColor(nrg_raw, cv2.COLOR_BGR2RGB)

        # 3. Generowanie masek
        nrg_mask = detect_dead_trees_advanced(nrg)
        rgb_mask = generate_forest_mask(rgb)
        
        # 4. Łączenie masek (AND)
        combined_mask_pre = (nrg_mask > 0) & (rgb_mask > 0)
        combined_mask = remove_small_objects(combined_mask_pre, min_size=config.MIN_OBJECT_SIZE).astype(np.uint8)
        
        # 5. Obliczenia
        iou_nrg = calculate_iou(nrg_mask, gt_mask)
        iou_rgb = calculate_iou(rgb_mask, gt_mask)
        iou_combined = calculate_iou(combined_mask, gt_mask)
        
        tp, tn, fp, fn = calculate_confusion_matrix_components(combined_mask, gt_mask)
        tp_nrg, tn_nrg, fp_nrg, fn_nrg = calculate_confusion_matrix_components(nrg_mask, gt_mask)
        tp_rgb, tn_rgb, fp_rgb, fn_rgb = calculate_confusion_matrix_components(rgb_mask, gt_mask)
        
        results.append({
            "rgb": rgb, "nrg": nrg,
            "nrg_mask": nrg_mask, "rgb_mask": rgb_mask, "combined_mask": combined_mask * 255, 
            "original_mask": gt_mask,
            "iou_nrg": iou_nrg, "iou_rgb": iou_rgb, "iou_combined": iou_combined,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "tp_nrg": tp_nrg, "tn_nrg": tn_nrg, "fp_nrg": fp_nrg, "fn_nrg": fn_nrg,
            "tp_rgb": tp_rgb, "tn_rgb": tn_rgb, "fp_rgb": fp_rgb, "fn_rgb": fn_rgb,
            "filename": os.path.basename(rgb_path)
        })
        
        if output_dir:
            cv2.imwrite(os.path.join(output_dir, f'Result_{os.path.basename(rgb_path)}'), combined_mask * 255)
            
    return results

# --- 4. WIZUALIZACJA I RAPORT ---
def plot_confusion_matrix(tp, tn, fp, fn, title):
    matrix = np.array([[tn, fp], [fn, tp]])
    total = tp + tn + fp + fn
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(matrix, cmap='Blues')
    for (i, j), val in np.ndenumerate(matrix):
        perc = val/total*100 if total > 0 else 0
        label = [['TN', 'FP'], ['FN', 'TP']][i][j]
        ax.text(j, i, f'{label}\n{val}\n({perc:.2f}%)', ha='center', va='center', color='black')
    plt.title(title)
    plt.ylabel('Rzeczywistość'); plt.xlabel('Predykcja'); plt.show()

def plot_iou_histogram(filenames, ious, title, color_bar):
    plt.figure(figsize=(12, 4))
    plt.bar(filenames, ious, color=color_bar)
    plt.title(title); plt.xticks(rotation=90); plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout(); plt.show()


# --- NOWA FUNKCJA TWORZĄCA OBRAZ RGB Z WARSTWAMI ---
def create_layered_image(gt_mask, pred_mask):
    """
    Ręczne tworzenie obrazu RGB:
    1. Tło = Ciemnoniebieski
    2. GT = Biały
    3. Pred = Czerwony (nadpisuje GT)
    """
    h, w = gt_mask.shape
    # 1. Tło: Ciemnoniebieski [0, 0, 100] (w formacie RGB)
    # Tworzymy puste płótno 3-kanałowe
    vis_img = np.full((h, w, 3), [0, 0, 100], dtype=np.uint8)
    
    # 2. Rysujemy GT na biało [255, 255, 255]
    # Wszędzie tam, gdzie maska GT > 0
    vis_img[gt_mask > 0] = [255, 255, 255]
    
    # 3. Rysujemy Predykcję na czerwono [255, 0, 0]
    # To nadpisze biały kolor tam, gdzie maski się pokrywają
    vis_img[pred_mask > 0] = [255, 0, 0]
    
    return vis_img

def show_1x3_blended(gt_mask, pred_mask, filename, pred_type):
    # Tworzymy obraz wizualizacyjny za pomocą nowej funkcji
    vis_img = create_layered_image(gt_mask, pred_mask)

    plt.figure(figsize=(18, 6))

    # 1. Maska Predykcji (LEWA)
    plt.subplot(1, 3, 1)
    plt.imshow(pred_mask, cmap='gray', vmin=0, vmax=255)
    plt.title(f"{pred_type}\n{filename[:15]}...")
    plt.axis('off')

    # 2. Maska GT (ŚRODEK)
    plt.subplot(1, 3, 2)
    plt.imshow(gt_mask, cmap='gray', vmin=0, vmax=255)
    plt.title("Wzorzec (Ground Truth)")
    plt.axis('off')

    # 3. Warstwy (PRAWA) - Bez cmap, bo to już jest gotowy obraz RGB
    plt.subplot(1, 3, 3)
    plt.imshow(vis_img)
    plt.title(f"Nałożenie: {pred_type} (Czerwony) na GT (Biały)\nNiebieski=Tło")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def display_full_report(results):
    # 1. Raport podstawowy (6 kolumn)
    print("--- RAPORT PODSTAWOWY ---")
    for r in results:
        plt.figure(figsize=(24, 5)) 
        plt.subplot(1, 6, 1); plt.imshow(r["rgb"]); plt.title(f"RGB: {r['filename'][:8]}..."); plt.axis('off')
        plt.subplot(1, 6, 2); plt.imshow(r["nrg"]); plt.title("Obraz NRG"); plt.axis('off')
        plt.subplot(1, 6, 3); plt.imshow(r["nrg_mask"], cmap='gray'); plt.title(f"Maska NRG\nIoU: {r['iou_nrg']:.2f}"); plt.axis('off')
        plt.subplot(1, 6, 4); plt.imshow(r["rgb_mask"], cmap='gray'); plt.title(f"Maska Lasu (RGB)\nIoU: {r['iou_rgb']:.2f}"); plt.axis('off')
        plt.subplot(1, 6, 5); plt.imshow(r["combined_mask"], cmap='gray'); plt.title(f"Combined\nIoU: {r['iou_combined']:.2f}"); plt.axis('off')
        plt.subplot(1, 6, 6); plt.imshow(r["original_mask"], cmap='gray', vmin=0, vmax=255); plt.title("Ground Truth"); plt.axis('off')
        plt.tight_layout(); plt.show()

    # 2. Histogramy
    filenames = [r["filename"] for r in results]
    plot_iou_histogram(filenames, [r["iou_nrg"] for r in results], "IoU dla maski NRG", "skyblue")
    plot_iou_histogram(filenames, [r["iou_rgb"] for r in results], "IoU dla maski RGB", "lightgreen")
    plot_iou_histogram(filenames, [r["iou_combined"] for r in results], "IoU dla maski Combined", "purple")

    # 3. Metryki zbiorcze
    def summarize_metrics(key_suffix, title):
        total_tp = sum(r[f'tp{key_suffix}'] for r in results)
        total_tn = sum(r[f'tn{key_suffix}'] for r in results)
        total_fp = sum(r[f'fp{key_suffix}'] for r in results)
        total_fn = sum(r[f'fn{key_suffix}'] for r in results)
        
        if key_suffix == "": target_iou_key = "iou_combined"
        else: target_iou_key = f'iou{key_suffix}'
            
        mean_iou = np.mean([r[target_iou_key] for r in results])
        print(f"\n--- WYNIKI DLA: {title} ---")
        print(f"Mean IoU: {mean_iou:.4f}")
        plot_confusion_matrix(total_tp, total_tn, total_fp, total_fn, f"Macierz Pomyłek: {title}")

    summarize_metrics("_nrg", "Maska NRG")
    summarize_metrics("_rgb", "Maska RGB (Las)")
    summarize_metrics("", "Maska Combined")
    
    # 4. ZESTAWIENIA SZCZEGÓŁOWE 1x3
    print("\n--- SZCZEGÓŁOWE PORÓWNANIE Z WZORCEM (GT) ---")
    for r in results:
        # Zestawienie dla maski RGB
        show_1x3_blended(r["original_mask"], r["rgb_mask"], r["filename"], "Maska RGB (Las)")
        
        # Zestawienie dla maski Combined
        show_1x3_blended(r["original_mask"], r["combined_mask"], r["filename"], "Maska Combined")


# --- MAIN ---
if __name__ == '__main__':
    paired = create_paired_files(config.PATHS["RGB"], config.PATHS["NRG"], config.PATHS["MASKS"])
    if paired:
        results = compare_and_score_masks(paired, num_samples=config.NUM_SAMPLES_TO_PROCESS, output_dir=config.PATHS["OUTPUT"])
        display_full_report(results)
    else:
        print("BŁĄD: Nie znaleziono par plików!")