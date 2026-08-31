# ================================================================
# ndsi_baseline.py
# Run this BEFORE the model is ready
# Computes NDSI baseline areas for all years
# ================================================================

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR    = Path("outputs/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS      = list(range(2017, 2024))
PIXEL_SIZE = 10  # metres

def ndsi_baseline(image, threshold=0.6):
    """
    Threshold on normalized NDSI channel (channel 0).
    0.6 in normalized [0,1] space ≈ NDSI > 0.4 in raw space.
    """
    ndsi_channel = image[:, :, 0]
    return (ndsi_channel > threshold).astype(np.uint8)

def compute_area_km2(mask, pixel_size_m=10):
    return mask.sum() * (pixel_size_m ** 2) / 1e6

ndsi_areas = []
ndsi_masks = []

for year in YEARS:
    path = PROCESSED_DIR / f"image_{year}.npy"
    if not path.exists():
        print(f"Missing {year}, skipping")
        continue

    image     = np.load(path)          # (H, W, 3)
    mask      = ndsi_baseline(image)
    area      = compute_area_km2(mask)
    ndsi_areas.append(area)
    ndsi_masks.append(mask)
    print(f"{year}: NDSI glacier area = {area:.2f} km²")

ndsi_areas = np.array(ndsi_areas)
years_arr  = np.array(YEARS[:len(ndsi_areas)])

# ---- Quick sanity plot ----
plt.figure(figsize=(10, 5))
plt.plot(years_arr, ndsi_areas, 's--', color='orange',
         linewidth=2, markersize=8, label='NDSI Baseline')
plt.xlabel("Year", fontsize=13)
plt.ylabel("Glacier Area (km²)", fontsize=13)
plt.title("Gangotri Glacier — NDSI Baseline Area (2017–2023)", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ndsi_baseline_area.png", dpi=150)
plt.show()

# Save for later use in full analysis
np.save(PROCESSED_DIR / "ndsi_areas.npy",  ndsi_areas)
np.save(PROCESSED_DIR / "years_arr.npy",   years_arr)
np.save(PROCESSED_DIR / "ndsi_masks.npy",  np.array(ndsi_masks))

print("\nNDSI baseline done. Results saved.")
print(f"Total area loss (NDSI): {ndsi_areas[0] - ndsi_areas[-1]:.2f} km²")
print(f"Average annual retreat: {(ndsi_areas[0] - ndsi_areas[-1]) / 6:.2f} km²/year")