import numpy as np
from PIL import Image, ImageFilter, ImageStat

def validate_leaf(image_path):
    """
    Validates whether an uploaded image contains a real agricultural leaf
    or an artificial/non-leaf object (e.g. cat, dog, person, car, flat green wall,
    green cloth, green plastic, neon graphics, paper, or an animal/person outdoors
    in front of a blurry grassy/garden background).

    Returns:
        is_valid (bool): True if image passes leaf characteristics, False otherwise.
        reason (str): Description if rejected, or 'OK'.
    """
    try:
        with Image.open(image_path) as pil_img:
            img_rgb = pil_img.convert("RGB")
            # Resize for fast, consistent analysis
            img_resized = img_rgb.resize((256, 256))
            arr = np.array(img_resized, dtype=np.float32)

        # ------------------------------------------------------------------
        # 1. Biological Foliage Green & Plant Hue Ratio Check
        # ------------------------------------------------------------------
        # Real crop leaves contain living plant chlorophyll and foliage.
        # Non-plant images without plants (indoors, cars, documents, pets indoors)
        # have virtually 0% green foliage (< 1%).
        img_hsv = img_resized.convert("HSV")
        hsv_arr = np.array(img_hsv, dtype=np.float32)
        h = hsv_arr[:, :, 0] * (360.0 / 255.0)  # Scale to 0-360 degrees
        s = hsv_arr[:, :, 1] / 255.0           # Scale to 0-1
        v = hsv_arr[:, :, 2] / 255.0           # Scale to 0-1

        # Plant foliage green & yellow-green range:
        green_foliage_mask = (h >= 35) & (h <= 165) & (s >= 0.08) & (v >= 0.08)
        green_foliage_ratio = float(np.mean(green_foliage_mask))

        if green_foliage_ratio < 0.035:
            return False, "No crop leaf foliage detected. Please upload a clear photo of a plant leaf."

        # ------------------------------------------------------------------
        # 2. Texture & Flat Surface Check (Laplacian / Edge Variance)
        # ------------------------------------------------------------------
        # Real leaves have biological venation (veins), serrations, texture,
        # or disease lesions (edge_mean typically 11 - 65, edge_var > 600).
        gray = img_resized.convert("L")
        edges_img = gray.filter(ImageFilter.FIND_EDGES)
        edges_arr = np.array(edges_img, dtype=np.float32)
        edge_stat = ImageStat.Stat(edges_img)
        edge_mean = edge_stat.mean[0]   # Overall edge intensity
        edge_var = edge_stat.var[0]     # Variance in edge intensity

        # Color standard deviation across RGB channels
        std_r = np.std(arr[:, :, 0])
        std_g = np.std(arr[:, :, 1])
        std_b = np.std(arr[:, :, 2])
        color_std_avg = float((std_r + std_g + std_b) / 3.0)

        # Completely flat green wall, paper, or uniform digital banner:
        if color_std_avg < 10.0 and edge_mean < 4.0:
            return False, "Flat or textureless surface detected (e.g. wall, paper, or uniform cloth)."

        if edge_mean < 3.0:
            return False, "Textureless or empty surface detected. Please upload a clear photo of a plant leaf."

        # ------------------------------------------------------------------
        # 3. Out-of-Focus Green Backdrop Check (e.g. Animal/Person in garden/grass)
        # ------------------------------------------------------------------
        # When an animal, pet (e.g. cat/dog), person, or object is photographed outdoors,
        # the camera lens focuses on the subject, leaving the background grass/plants
        # blurred (bokeh). In contrast, in genuine leaf photos, the leaf itself is in focus.
        if np.any(green_foliage_mask):
            green_edges = edges_arr[green_foliage_mask]
            gem = float(np.mean(green_edges)) if len(green_edges) > 0 else 0.0

            non_green_edges = edges_arr[~green_foliage_mask]
            ngem = float(np.mean(non_green_edges)) if len(non_green_edges) > 0 else 0.0

            # If the green portion is blurry background (gem < 10.0)
            # while the non-green foreground subject is in sharp focus (ngem > 18.0 and > 2x gem):
            if gem < 10.0 and ngem > 18.0 and ngem > 2.0 * gem:
                return False, "Non-leaf subject detected in front of plant background (e.g. animal or object outdoors)."

            # Subject Centricity Check:
            # When a farmer photographs a leaf for disease diagnosis, the leaf is centered.
            # If the center 50% box is occupied by a non-leaf subject (< 40% green) with high non-green edges:
            center_green = float(np.mean(green_foliage_mask[64:192, 64:192]))
            if center_green < 0.40 and ngem > 15.0:
                return False, "Non-leaf object detected in center. Please focus directly on the crop leaf."

        # ------------------------------------------------------------------
        # 4. Synthetic / Neon Artificial Green Detection
        # ------------------------------------------------------------------
        # Natural plant leaves contain chlorophyll, carotenoids, and water.
        # Artificial neon green (highlighters, chroma key screens, neon plastics)
        # has unnatural green dominance: G >> R + B with extreme saturation.
        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]

        green_dominant_mask = (g > 50) & (g > r) & (g > b)
        if np.any(green_dominant_mask):
            g_dom = g[green_dominant_mask]
            r_dom = r[green_dominant_mask]
            b_dom = b[green_dominant_mask]

            # Neon / digital green ratio: G / (R + B + 1)
            neon_ratios = g_dom / (r_dom + b_dom + 1e-4)
            high_neon_fraction = float(np.mean(neon_ratios > 2.8))

            if high_neon_fraction > 0.40:
                return False, "Synthetic or neon green artificial object detected."

        # ------------------------------------------------------------------
        # 5. Smooth Synthetic Object Interior Check (e.g. green cup, toy, bottle)
        # ------------------------------------------------------------------
        # Solid smooth green plastic objects have high green ratio but very low
        # internal edge density and low edge variance compared to leaves.
        if np.any(green_dominant_mask):
            green_edges = edges_arr[green_dominant_mask]
            internal_edge_density = float(np.mean(green_edges > 15.0))

            if green_foliage_ratio > 0.40 and internal_edge_density < 0.008 and edge_var < 40.0:
                return False, "Smooth synthetic green object detected (lacks leaf venation texture)."

        return True, "OK"

    except Exception as e:
        # If image cannot be read or processed, fail safe
        return False, f"Image processing error: {str(e)}"
