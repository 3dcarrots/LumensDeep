# Ultra-High Resolution Global Optical Matting: Guided Upsampling Algorithm

## 1. Introduction and Problem Statement

Creating a perfect alpha mask (Subject Mask) for ultra-high resolution photographs (up to 40 megapixels / 8K) is one of the most difficult challenges in modern computer vision. Traditional neural network models are optimized to work with relatively low resolutions (typically 1024x1024 pixels) due to strict video memory (VRAM) limitations.

When directly processing an 8K image with such models, an inevitable downscaling at the input and bilinear interpolation (upsampling) at the output occurs. This results in a semantically correct but optically blurred mask, where all high-frequency microstructures are lost: individual hairs, pores of translucent fabrics (nets, veils), and diffractive edge blurring.

Attempts to solve this problem through spatial fragmentation (slicing the image into 1024x1024 patches) lead to the loss of global semantic context: the neural network may fail to recognize an isolated fragment of a dress as part of the foreground, causing "black holes" in the mask and visible seams at the boundaries of the patches. Using Depth Maps for clipping is also ineffective because depth does not capture the sub-pixel transparency of individual hairs.

## 2. Two-Stage Matting Architecture (Hybrid Pipeline)

To overcome these fundamental limitations, our pipeline implements a hybrid two-stage algorithm that separates the tasks of semantic recognition and optical edge reconstruction. This approach allows us to obtain an uncompromising 8K mask in fractions of a second, bypassing the fragmentation problem.

### Stage 1: Low-Frequency Semantic Localization

In the first stage, the original 8K image is fed into a neural network model. The model performs global semantic analysis at a reduced resolution.
The result is a "soft" mask $\alpha_{base}$, which perfectly understands *where* the object is located but has blurred edges at the micro-level.

To prepare this mask for mathematical optimization, a nonlinear tonal correction is applied:
1. **Black Point Shift:** $(\alpha_{base} - 0.1) / 0.8$ — this expands the zone of guaranteed background and guaranteed object, narrowing the zone of uncertainty (Unknown Region) exclusively to the physical boundaries of the object.
2. **Gamma Correction:** $\alpha_{base}^{0.5}$ — boosts the midtones, ensuring that the "core" of the object (its inner part) becomes perfectly white (1.0), preventing unwanted transparency of dense materials.

### Stage 2: High-Frequency Mathematical Scaling (High-Resolution Fix)

The key innovation of the algorithm is the use of the **Guided Image Filter** to mathematically transfer microstructures from the original 40-megapixel photograph onto the blurred semantic mask.

In this process, the original color RGB image is converted into a high-contrast grayscale image (Gray Guide, denoted as $I$). The blurred semantic mask acts as the target signal ($p$).

The Guided Filter is based on the assumption of a local linear model. It postulates that the final, perfectly sharp mask ($q$) is a linear transformation of the guidance image ($I$) within a small local window $w_k$ of radius $r$:
$$q_i = a_k I_i + b_k, \quad \forall i \in w_k$$

The coefficients $a_k$ and $b_k$ are calculated analytically by minimizing the difference between the input blurred mask $p$ and the output $q$. This involves calculating the variance of the pixels of the guidance image ($\sigma_k^2$) and their covariance with the blurred mask:
$$a_k = \frac{\text{cov}(I, p)_k}{\sigma_k^2 + \epsilon}$$
$$b_k = \bar{p}_k - a_k \bar{I}_k$$

Where $\epsilon$ is the regularization parameter.

## 3. Configuration and Physical Meaning of Parameters

To achieve perfect hair extraction, the algorithm uses specifically selected hyperparameters:
- **Window Radius ($r = 20$):** A sufficiently large radius that completely covers the blur width from bilinear interpolation. It ensures that the local window captures both the guaranteed background and the guaranteed object.
- **Regularization Parameter ($\epsilon = 1e-3$):** An extremely low value of $\epsilon$ forces the filter to aggressively follow the smallest gradient fluctuations in the guidance image. Since every single hair in the original 8K photo represents a sharp brightness transition (high variance $\sigma_k^2$), the coefficient $a_k$ approaches 1, resulting in the direct copying of the hair structure into the alpha mask.

## 4. Advantages and Results

By shifting from local fragmentation (patches) to global optical mathematics, the following results have been achieved:
1. **Global Integrity:** Seam artifacts and "false positives" are eliminated because the optical math operates only in areas where the base mask (from the neural network) created a gradient.
2. **Microstructure Accuracy:** Hair in the wind, holes in lace, and translucent edges are rendered with sub-pixel 1:1 accuracy relative to the original 8K file, as the filter extracts them directly from the RGB pixels.
3. **Extreme Performance:** The calculation of variance and covariance is reduced to a series of fast integral convolutions (Fast Box Blur / `cv2.blur`). This allows the algorithm to be applied to the entire 40-megapixel canvas simultaneously in fractions of a second, making it suitable for real-time interactive use.

This architecture proves that the synergy of modern semantic neural networks (for macro-localization) and classical analytical matting (for micro-reconstruction) is the most efficient path for processing ultra-high resolution images in VFX and computer graphics.
