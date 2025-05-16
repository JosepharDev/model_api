import os
import ee
import geemap
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import urllib.request
from segment_anything import SamPredictor, sam_model_registry

# === Step 0: Download SAM model if not present ===
SAM_CHECKPOINT = "sam_vit_l.pth"
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"

if not os.path.exists(SAM_CHECKPOINT):
    print("Downloading SAM model...")
    urllib.request.urlretrieve(SAM_URL, SAM_CHECKPOINT)
    print("Download complete.")

# === Step 1: Initialize Earth Engine ===
ee.Authenticate()  # optional in some Docker cases if already authenticated
ee.Initialize(project='certain-catcher-430110-v2')  # replace with your GEE project ID

# === Step 2: Choose coordinates and region ===
lat, lon = 32.512928, -8.816545
point = ee.Geometry.Point([lon, lat])
buffer = point.buffer(400)
region = buffer.bounds()

# === Step 3: Get Sentinel-2 image ===
s2 = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(buffer)
    .filterDate("2019-03-01", "2019-03-31")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
    .select(["B4", "B3", "B2"])
    .median()
    .clip(buffer)
)

# === Step 4: Convert to NumPy ===
rgb_array = geemap.ee_to_numpy(s2, region=region, scale=10)

if rgb_array.shape[-1] != 3:
    raise ValueError(f"Expected 3-band image, got shape: {rgb_array.shape}")

# === Step 5: Preprocess for SAM ===
rgb_array = np.nan_to_num(rgb_array)
rgb_array = np.clip(rgb_array / 3000 * 255, 0, 255).astype(np.uint8)
rgb_image = rgb_array

# Save optional image preview
Image.fromarray(rgb_image).save("field_rgb.png")

# === Step 6: Convert lat/lon to pixel ===
def latlon_to_pixel_coords(lat, lon, region_bounds, img_width, img_height):
    coords = region_bounds["coordinates"][0]
    lon_min = min(c[0] for c in coords)
    lon_max = max(c[0] for c in coords)
    lat_min = min(c[1] for c in coords)
    lat_max = max(c[1] for c in coords)

    x_ratio = (lon - lon_min) / (lon_max - lon_min)
    y_ratio = (lat_max - lat) / (lat_max - lat_min)

    px = int(x_ratio * img_width)
    py = int(y_ratio * img_height)

    return [px, py]

xy = latlon_to_pixel_coords(lat, lon, region.getInfo(), rgb_image.shape[1], rgb_image.shape[0])
input_point = np.array([xy])
input_label = np.array([1])

# === Step 7: Load and run SAM ===
sam = sam_model_registry["vit_l"](checkpoint=SAM_CHECKPOINT)
sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
predictor = SamPredictor(sam)
predictor.set_image(rgb_image)

masks, scores, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

# === Step 8: Show or Save Masks ===
for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure()
    plt.imshow(rgb_image)
    plt.imshow(mask, alpha=0.5, cmap='Reds')
    plt.title(f"Mask {i+1} (Score: {score:.3f})")
    plt.axis("off")
    plt.savefig(f"mask_{i+1}.png")
    plt.close()
