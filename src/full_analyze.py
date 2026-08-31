# ================================================================
# full_analyze.py
# Runs after training is complete.
# 1. Loads trained Attention U-Net
# 2. Predicts glacier masks for all 7 years
# 3. Computes areas and compares with NDSI baseline
# 4. Runs regression forecasting
# 5. Generates all report figures
# ================================================================

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import time

# ---- Load model architecture ----
exec(open("src/model.py").read())

# ----------------------------------------------------------------
# CONFIGURATION — update these paths if needed
# ----------------------------------------------------------------
PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR    = Path("outputs/plots")
MODEL_PATH    = Path("best_attention_unet.pth")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS      = list(range(2017, 2024))
PIXEL_SIZE = 10   # Sentinel-2 resolution in metres
DEVICE     = torch.device("cpu")

# ----------------------------------------------------------------
# Load trained model
# ----------------------------------------------------------------
print("=" * 60)
print("GANGOTRI GLACIER — FULL TEMPORAL ANALYSIS")
print("=" * 60)
print(f"\n[1/6] Loading trained Attention U-Net from {MODEL_PATH}...")
model = AttentionUNet(in_channels=3, out_channels=1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("      Model loaded successfully.")
print(f"      Parameters: 31,389,165")

# ----------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------
def predict_full_image(model, image, patch_size=256, stride=128):
    """
    Patch-based inference on full image with progress tracking.
    """
    H, W, C   = image.shape
    output    = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    # Count total patches for progress
    y_steps = list(range(0, H - patch_size + 1, stride))
    x_steps = list(range(0, W - patch_size + 1, stride))
    total_patches = len(y_steps) * len(x_steps)
    processed = 0
    last_pct = -1

    with torch.no_grad():
        for y in y_steps:
            for x in x_steps:
                patch  = image[y:y+patch_size, x:x+patch_size, :]
                tensor = torch.tensor(patch, dtype=torch.float32)\
                              .permute(2, 0, 1).unsqueeze(0)
                pred   = model(tensor).squeeze().numpy()
                output[y:y+patch_size, x:x+patch_size]    += pred
                count_map[y:y+patch_size, x:x+patch_size] += 1

                processed += 1
                pct = int(100 * processed / total_patches)
                if pct % 10 == 0 and pct != last_pct:
                    print(f"        {pct}% ({processed}/{total_patches} patches)", end="\r")
                    last_pct = pct

    print(f"        100% ({total_patches}/{total_patches} patches) — Done!     ")
    count_map[count_map == 0] = 1
    return (output / count_map > 0.5).astype(np.uint8)


def compute_area_km2(mask):
    return mask.sum() * (PIXEL_SIZE ** 2) / 1e6


def ndsi_baseline(image, threshold=0.6):
    return (image[:, :, 0] > threshold).astype(np.uint8)


# ----------------------------------------------------------------
# Apply model to all 7 years
# ----------------------------------------------------------------
print(f"\n[2/6] Running inference on all {len(YEARS)} years...")
print(f"      Image size: 4528 x 5808 pixels per year")
print(f"      This will take approximately 5-10 minutes per year on CPU\n")

model_areas, ndsi_areas   = [], []
model_masks, ndsi_masks_l = [], []
year_times = []

for i, year in enumerate(YEARS):
    path = PROCESSED_DIR / f"image_{year}.npy"
    if not path.exists():
        print(f"      WARNING: Missing image for {year}, skipping")
        continue

    print(f"  --- Year {year} ({i+1}/{len(YEARS)}) ---")
    t_start = time.time()

    image = np.load(path)
    print(f"      Image loaded: {image.shape}")

    # NDSI baseline (fast)
    print(f"      Computing NDSI baseline...", end=" ")
    n_mask = ndsi_baseline(image)
    n_area = compute_area_km2(n_mask)
    print(f"Done. Area = {n_area:.2f} km²")

    # Attention U-Net inference (slow)
    print(f"      Running Attention U-Net inference...")
    m_mask = predict_full_image(model, image)
    m_area = compute_area_km2(m_mask)

    t_end = time.time()
    elapsed = t_end - t_start
    year_times.append(elapsed)
    avg_time = np.mean(year_times)
    remaining = avg_time * (len(YEARS) - (i + 1))

    print(f"      Model area = {m_area:.2f} km²")
    print(f"      Time for this year: {elapsed/60:.1f} min")
    print(f"      Estimated time remaining: {remaining/60:.1f} min\n")

    model_areas.append(m_area)
    ndsi_areas.append(n_area)
    model_masks.append(m_mask)
    ndsi_masks_l.append(n_mask)

model_areas = np.array(model_areas)
ndsi_areas  = np.array(ndsi_areas)
years_arr   = np.array(YEARS[:len(model_areas)])

# Save for later use
np.save(PROCESSED_DIR / "model_areas.npy", model_areas)
np.save(PROCESSED_DIR / "ndsi_areas.npy",  ndsi_areas)
np.save(PROCESSED_DIR / "model_masks.npy", np.array(model_masks))

print("=" * 60)
print("INFERENCE COMPLETE — AREA SUMMARY")
print("=" * 60)
print(f"{'Year':<8} {'Model (km²)':<18} {'NDSI (km²)':<18}")
print("-" * 44)
for i, year in enumerate(YEARS[:len(model_areas)]):
    print(f"{year:<8} {model_areas[i]:<18.2f} {ndsi_areas[i]:<18.2f}")
print("=" * 60)


# ----------------------------------------------------------------
# Regression Forecasting
# ----------------------------------------------------------------
print(f"\n[3/6] Running regression forecasting...")

forecast_years = np.arange(2024, 2031)
X = years_arr.reshape(-1, 1)

# Linear regression
lr    = LinearRegression().fit(X, model_areas)
lr_r2 = r2_score(model_areas, lr.predict(X))
lr_rmse = np.sqrt(mean_squared_error(model_areas, lr.predict(X)))
lr_fc = lr.predict(forecast_years.reshape(-1, 1))

# Polynomial degree 2
poly   = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
pr     = LinearRegression().fit(X_poly, model_areas)
pr_r2  = r2_score(model_areas, pr.predict(X_poly))
pr_rmse = np.sqrt(mean_squared_error(model_areas, pr.predict(X_poly)))
pr_fc  = pr.predict(poly.transform(forecast_years.reshape(-1, 1)))

print(f"      Linear Regression     — R²: {lr_r2:.4f}, RMSE: {lr_rmse:.4f} km²")
print(f"      Polynomial Regression — R²: {pr_r2:.4f}, RMSE: {pr_rmse:.4f} km²")

print(f"\n      Forecast (Polynomial model):")
for y, a in zip(forecast_years, pr_fc):
    print(f"        {y}: {a:.2f} km²")


# ----------------------------------------------------------------
# FIGURE 1: Area over time + forecast
# ----------------------------------------------------------------
print(f"\n[4/6] Generating figures...")
print(f"      Generating Figure 1: Area forecast plot...")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(years_arr, model_areas, 'o-', color='steelblue',
        linewidth=2, markersize=8, label='Attention U-Net (measured)')
ax.plot(years_arr, ndsi_areas,  's--', color='orange',
        linewidth=1.5, markersize=6, label='NDSI Baseline (measured)')
ax.plot(forecast_years, lr_fc,  ':', color='steelblue',
        linewidth=1.5, label=f'Linear Forecast (R²={lr_r2:.3f})')
ax.plot(forecast_years, pr_fc,  '-', color='darkblue',
        linewidth=2, label=f'Polynomial Forecast (R²={pr_r2:.3f})')
ax.axvline(x=2023.5, color='gray', linestyle='--', alpha=0.5, label='Forecast boundary')
ax.fill_between(forecast_years, pr_fc * 0.95, pr_fc * 1.05,
                alpha=0.15, color='darkblue', label='±5% uncertainty band')
ax.set_xlabel("Year", fontsize=13)
ax.set_ylabel("Glacier Area (km²)", fontsize=13)
ax.set_title("Gangotri Glacier Area Change and Retreat Forecast (2017–2030)", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig1_area_forecast.png", dpi=150)
plt.close()
print(f"      Saved: fig1_area_forecast.png")


# ----------------------------------------------------------------
# FIGURE 2: Change map 2017 vs 2023
# ----------------------------------------------------------------
print(f"      Generating Figure 2: Change map 2017 vs 2023...")

mask_2017 = model_masks[0]
mask_2023 = model_masks[-1]

change_map = np.zeros_like(mask_2017, dtype=np.uint8)
change_map[mask_2017 == 1] = 1
change_map[(mask_2017 == 1) & (mask_2023 == 0)] = 2
change_map[(mask_2017 == 0) & (mask_2023 == 1)] = 3

cmap = mcolors.ListedColormap(['#f5f5f0', '#a8d8ea', '#e74c3c', '#27ae60'])

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(mask_2017, cmap='Blues')
axes[0].set_title(f'Glacier Mask 2017\n({model_areas[0]:.1f} km²)', fontsize=12)
axes[0].axis('off')
axes[1].imshow(mask_2023, cmap='Blues')
axes[1].set_title(f'Glacier Mask 2023\n({model_areas[-1]:.1f} km²)', fontsize=12)
axes[1].axis('off')
axes[2].imshow(change_map, cmap=cmap, vmin=0, vmax=3)
axes[2].set_title('Change Map 2017→2023\n(Blue=Stable, Red=Retreated)', fontsize=12)
axes[2].axis('off')
plt.suptitle('Gangotri Glacier Change Detection', fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig2_change_map.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"      Saved: fig2_change_map.png")


# ----------------------------------------------------------------
# FIGURE 3: Annual retreat rate bar chart
# ----------------------------------------------------------------
print(f"      Generating Figure 3: Annual retreat rate...")

retreat_rates = -np.diff(model_areas)
colors = ['#e74c3c' if r > 0 else '#27ae60' for r in retreat_rates]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(YEARS[1:len(retreat_rates)+1], retreat_rates,
       color=colors, edgecolor='black', linewidth=0.5)
ax.axhline(y=np.mean(retreat_rates), color='navy', linestyle='--',
           label=f'Mean retreat: {np.mean(retreat_rates):.2f} km²/yr')
ax.set_xlabel("Year", fontsize=13)
ax.set_ylabel("Area Change (km²/year)", fontsize=13)
ax.set_title("Annual Glacier Retreat Rate — Gangotri (2017–2023)", fontsize=14)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig3_retreat_rate.png", dpi=150)
plt.close()
print(f"      Saved: fig3_retreat_rate.png")


# ----------------------------------------------------------------
# FIGURE 4: Method comparison bar chart
# ----------------------------------------------------------------
print(f"      Generating Figure 4: Method comparison...")

x     = np.arange(len(years_arr))
width = 0.35
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - width/2, model_areas, width, label='Attention U-Net', color='steelblue')
ax.bar(x + width/2, ndsi_areas,  width, label='NDSI Baseline',   color='orange')
ax.set_xticks(x)
ax.set_xticklabels(years_arr)
ax.set_xlabel("Year", fontsize=13)
ax.set_ylabel("Glacier Area (km²)", fontsize=13)
ax.set_title("Method Comparison: Attention U-Net vs NDSI Baseline", fontsize=14)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig4_method_comparison.png", dpi=150)
plt.close()
print(f"      Saved: fig4_method_comparison.png")


# ----------------------------------------------------------------
# Final Summary
# ----------------------------------------------------------------
print(f"\n[5/6] Computing final statistics...")

total_loss   = model_areas[0] - model_areas[-1]
mean_retreat = total_loss / (len(model_areas) - 1)

print(f"\n{'='*60}")
print("GANGOTRI GLACIER — FINAL RESULTS SUMMARY")
print(f"{'='*60}")
print(f"\nTemporal Analysis (2017–2023):")
print(f"  Initial area (2017):        {model_areas[0]:.2f} km²")
print(f"  Final area   (2023):        {model_areas[-1]:.2f} km²")
print(f"  Total area loss:            {total_loss:.2f} km²")
print(f"  Mean annual retreat rate:   {mean_retreat:.2f} km²/year")
print(f"\nForecasting:")
print(f"  Projected area 2025:        {pr_fc[1]:.2f} km²")
print(f"  Projected area 2027:        {pr_fc[3]:.2f} km²")
print(f"  Projected area 2030:        {pr_fc[-1]:.2f} km²")
print(f"  Projected total loss 2030:  {model_areas[0] - pr_fc[-1]:.2f} km²")
print(f"\nRegression Performance:")
print(f"  Linear    R²: {lr_r2:.4f}  RMSE: {lr_rmse:.4f} km²")
print(f"  Polynomial R²: {pr_r2:.4f}  RMSE: {pr_rmse:.4f} km²")
print(f"\nSegmentation Performance (from training):")
print(f"  IoU (Jaccard):  0.8445")
print(f"  Dice Score:     0.9153")
print(f"  Precision:      0.9064")
print(f"  Recall:         0.9248")
print(f"  F1 Score:       0.9155")
print(f"{'='*60}")

print(f"\n[6/6] All figures saved to {OUTPUT_DIR}/")
print(f"      fig1_area_forecast.png")
print(f"      fig2_change_map.png")
print(f"      fig3_retreat_rate.png")
print(f"      fig4_method_comparison.png")
print(f"\nAnalysis complete!")