# ================================================================
# preprocess.py
# Reads GEE exports + RGI shapefile, creates training patches
# ================================================================

import os
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import geopandas as gpd
from pathlib import Path
from tqdm import tqdm

# ----------------------------------------------------------------
# CONFIGURATION — adjust paths if needed
# ----------------------------------------------------------------
RAW_DIR       = Path("data/raw")          # GeoTIFF files from GEE
RGI_PATH      = Path("data/rgi/14_rgi60_CentralAsia.shp")
PROCESSED_DIR = Path("data/processed")
PATCH_SIZE    = 256   # pixels
STRIDE        = 128   # overlap = PATCH_SIZE - STRIDE = 128px
YEARS         = list(range(2017, 2024))

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
(PROCESSED_DIR / "images").mkdir(exist_ok=True)
(PROCESSED_DIR / "masks").mkdir(exist_ok=True)

# ----------------------------------------------------------------
# STEP A: Load RGI shapefile and filter to Gangotri region
# ----------------------------------------------------------------
print("Loading RGI shapefile...")
rgi = gpd.read_file(RGI_PATH)

# Gangotri approximate bounding box
gangotri_bbox = rgi.cx[78.90:79.50, 30.70:31.10]
print(f"Found {len(gangotri_bbox)} glacier polygons in Gangotri region")
print(gangotri_bbox[['RGIId', 'Name', 'Area']].head(10))  # inspect

# ----------------------------------------------------------------
# STEP B: Process each year's image
# ----------------------------------------------------------------
def normalize_band(band):
    """Normalize a single band to [0, 1] range"""
    b_min, b_max = band.min(), band.max()
    if b_max - b_min == 0:
        return np.zeros_like(band)
    return (band - b_min) / (b_max - b_min)


def rasterize_rgi(rgi_gdf, reference_raster_path):
    """
    Convert RGI vector polygons to a binary raster mask
    aligned with the reference Sentinel-2 image
    """
    with rasterio.open(reference_raster_path) as src:
        out_shape = (src.height, src.width)
        transform = src.transform
        
        # Reproject RGI to match the raster CRS
        rgi_proj = rgi_gdf.to_crs(src.crs)
        
        # Rasterize: glacier = 1, everything else = 0
        mask = rasterize(
            shapes=((geom, 1) for geom in rgi_proj.geometry),
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype=np.uint8
        )
    return mask


def extract_patches(image, mask, patch_size, stride):
    """
    Slide a window over the image and extract patches
    Returns only patches that have at least 5% glacier coverage
    (avoids training on patches that are entirely background)
    """
    h, w, _ = image.shape
    patches_img, patches_mask = [], []
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            img_patch  = image[y:y+patch_size, x:x+patch_size, :]
            mask_patch = mask[y:y+patch_size, x:x+patch_size]
            
            # Skip patches with NaN values (cloud-masked areas)
            if np.isnan(img_patch).any():
                continue
            
            # Keep patches with at least 1% glacier OR in training mode keep all
            glacier_fraction = mask_patch.mean()
            if glacier_fraction > 0.01 or np.random.rand() < 0.1:
                patches_img.append(img_patch)
                patches_mask.append(mask_patch)
    
    return patches_img, patches_mask


# Process the first available year for training mask rasterization
reference_year = 2020  # use a middle year as reference
ref_path = RAW_DIR / f"gangotri_{reference_year}.tif"

print(f"\nRasterizing RGI mask using {reference_year} image as reference...")
glacier_mask = rasterize_rgi(gangotri_bbox, ref_path)
print(f"Mask shape: {glacier_mask.shape}, Glacier pixels: {glacier_mask.sum()}")

# Save the mask for inspection
np.save(PROCESSED_DIR / "glacier_mask_rgi.npy", glacier_mask)


# Process all years
all_images, all_masks = [], []

for year in YEARS:
    tif_path = RAW_DIR / f"gangotri_{year}.tif"
    
    if not tif_path.exists():
        print(f"WARNING: {tif_path} not found, skipping year {year}")
        continue
    
    print(f"\nProcessing {year}...")
    
    with rasterio.open(tif_path) as src:
        # Read all 3 bands: NDSI (1), NDWI (2), NDVI (3)
        data = src.read()  # shape: (3, H, W)
        
    # Handle nodata/inf values from index computation
    data = np.where(np.isinf(data), np.nan, data)
    data = np.clip(data, -1, 1)  # indices are always in [-1, 1]
    
    # Normalize each band to [0, 1]
    ndsi = normalize_band(data[0])
    ndwi = normalize_band(data[1])
    ndvi = normalize_band(data[2])
    
    # Stack to H x W x 3
    image = np.stack([ndsi, ndwi, ndvi], axis=-1)
    
    # Save full-year image and mask for temporal analysis later
    np.save(PROCESSED_DIR / f"image_{year}.npy", image)
    np.save(PROCESSED_DIR / f"mask_{year}.npy", glacier_mask)
    
    # Extract training patches (only from years with RGI reference)
    imgs, masks = extract_patches(image, glacier_mask, PATCH_SIZE, STRIDE)
    all_images.extend(imgs)
    all_masks.extend(masks)
    print(f"  Extracted {len(imgs)} patches from {year}")


# ----------------------------------------------------------------
# STEP C: Train / Validation / Test Split (70 / 15 / 15)
# ----------------------------------------------------------------
print(f"\nTotal patches: {len(all_images)}")

all_images = np.array(all_images, dtype=np.float32)  # (N, 256, 256, 3)
all_masks  = np.array(all_masks,  dtype=np.float32)  # (N, 256, 256)

# Shuffle
indices = np.random.permutation(len(all_images))
all_images = all_images[indices]
all_masks  = all_masks[indices]

n = len(all_images)
n_train = int(0.70 * n)
n_val   = int(0.15 * n)

X_train = all_images[:n_train]
X_val   = all_images[n_train:n_train+n_val]
X_test  = all_images[n_train+n_val:]

y_train = all_masks[:n_train]
y_val   = all_masks[n_train:n_train+n_val]
y_test  = all_masks[n_train+n_val:]

# Save splits
np.save(PROCESSED_DIR / "X_train.npy", X_train)
np.save(PROCESSED_DIR / "X_val.npy",   X_val)
np.save(PROCESSED_DIR / "X_test.npy",  X_test)
np.save(PROCESSED_DIR / "y_train.npy", y_train)
np.save(PROCESSED_DIR / "y_val.npy",   y_val)
np.save(PROCESSED_DIR / "y_test.npy",  y_test)

print(f"\nSaved splits:")
print(f"  Train: {X_train.shape}")
print(f"  Val:   {X_val.shape}")
print(f"  Test:  {X_test.shape}")
print(f"\nPreprocessing complete! Upload data/processed/ folder to Kaggle.")