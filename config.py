import yaml
import numpy as np
import os

# Wczytywanie pliku YAML
with open("config.yaml", 'r') as stream:
    try:
        cfg = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Mapowanie ścieżek
PATHS = {
    "NRG": cfg['paths']['nrg_pattern'],
    "RGB": cfg['paths']['rgb_pattern'],
    "MASKS": cfg['paths']['masks_pattern'],
    "OUTPUT": cfg['paths']['output_dir']
}

# Parametry detekcji
NDVI_PERCENTILE = cfg['detection']['ndvi_percentile']
MIN_OBJECT_SIZE = cfg['detection']['min_object_size']
MORPH_KERNEL_SIZE = tuple(cfg['detection']['morph_kernel_size'])

# Progi HSV (konwertowane na numpy array dla OpenCV/Skimage)
FOREST_LOWER_PURPLE = np.array(cfg['forest_hsv']['lower'])
FOREST_UPPER_PURPLE = np.array(cfg['forest_hsv']['upper'])

# Ustawienia eksperymentu
NUM_SAMPLES_TO_PROCESS = cfg['experiment']['num_samples']