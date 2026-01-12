# Dead Tree Segmentation (NRG + RGB)

This project implements an advanced automated detection and segmentation system for standing dead trees (snags) using aerial multispectral (NRG) and high-definition (RGB) imagery. The system combines vegetation index analysis (NDVI/GNDVI) with forest canopy color filtration.

## 🚀 Key Features

- **Hybrid Segmentation**: Utilizes the Near-Infrared (NIR) channel from NRG images for necrotic tissue detection and HSV color space analysis for forest area isolation.
- **YAML Configuration**: Full management of algorithm parameters (thresholds, kernel sizes, file paths) via a centralized `config.yaml` file.
- **CLI Interface (Argparse)**: Ability to override any configuration parameter directly from the terminal at runtime.
- **Advanced Reporting**: Generates IoU statistics, confusion matrices, and comparative visualizations for performance evaluation.

# 🛠️ Virtual Environment & Dependencies
It is recommended to use a virtual environment to avoid dependency conflicts:

```bash
python -m venv .venv

## Activation (Windows):
.\.venv\Scripts\activate
## Activation (Linux/Mac):
source .venv/bin/activate

pip install -r requirements.txt
```
# Configuration Setup
The project requires a configuration file. Create it by copying the template:

```bash
cp config.temp.yaml config.yaml
```
Before running the code, please copy temp config into your own config

# 📂 Data Structure

The `data/` folder is ignored by version control. Please prepare your local data in the following structure:

```text
data/
└── USA_segmentation/
    ├── NRG_images/   # NIR-Red-Green imagery
    ├── RGB_images/   # High-definition RGB imagery
    └── masks/        # Ground Truth binary masks (PNG)
```
# Dataset
https://www.kaggle.com/datasets/meteahishali/aerial-imagery-for-standing-dead-tree-segmentation

# 🖥️ Usage (CLI)
The program offers a flexible Command Line Interface. Values entered in the terminal take precedence over those defined in config.yaml.

Default Execution:
```bash
python main.py
```

Quick Test (e.g., 2 samples with custom sensitivity):
```bash
python main.py --samples 2 --percentile 30
```

Custom Output Directory:
```bash
python main.py --output ./test_results_v1
```

For a full list of available flags, run:
```bash
python main.py --help.
```
# ⚙️ Configuration (config.yaml)

The following key parameters are defined in the config file:

- **ndvi_percentile**: Sensitivity threshold for NDVI/GNDVI indices.
- **forest_hsv**: Color ranges (Lower/Upper) used to identify the forest canopy.
- **min_object_size**: Minimum pixel count for an object to be retained (noise filtration).

# 📊 Visualization Interpretation (Seismic Blend)
The comparative visualization (third column of the 1x3 report) utilizes the seismic colormap to represent the logical agreement between the algorithm and the ground truth:

- **Dark Blue (Background)**: Correct identification of non-tree areas.
- **White (GT)**: Tree detected from Ground Truth Mask.
- **Dark Red (Combined_Mask)**: Tree detected by generated mask.

