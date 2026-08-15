# High-Resolution Depth Estimation and Normal Map Generation via Hierarchical Laplacian PatchFusion

## Abstract
Generating high-fidelity depth maps and normal maps from single 2D images is challenging due to the limited receptive fields and internal resolution constraints of Vision Transformer (ViT) based architectures like DepthAnything V2. When processing ultra-high-resolution images, naive tiling leads to visible grid seams, "pillowy" local distortions, and perspective distortions in normal map generation. This paper presents a novel algorithmic pipeline that employs Hierarchical Laplacian PatchFusion to seamlessly stitch depth patches while preserving global macro-geometry, alongside a mathematically optimized heightfield-to-normal conversion using non-linear depth squaring and spatial median filtering.

## 1. Introduction
Monocular depth estimation models such as DepthAnything V2 exhibit extraordinary relative depth perception but are constrained to fixed internal processing resolutions (e.g., 518x518 pixels). Processing larger images requires tiling, which introduces boundary artifacts and local shape distortions. Furthermore, converting these depth maps into tangent-space normal maps often yields overly contrastive or geometrically incorrect results due to the non-linear nature of disparity maps. We propose a comprehensive pipeline to solve these issues.

## 2. High-Resolution Depth Map Assembly

### 2.1. ViT Internal Resolution and Moiré Elimination
DepthAnything V2 internally processes images at a resolution of 518x518 pixels (a multiple of its 14x14 patch token size). A common pitfall in tiling implementations is using arbitrary patch sizes (e.g., 512x512). This forces the model to upscale the patch to 518x518 internally and downscale it back to 512x512, creating a sub-pixel misalignment that manifests as a massive Moiré pattern (horizontal and vertical interference stripes) across the depth map. By strictly enforcing a patch size of exactly 518x518, we achieve 1:1 pixel mapping, completely bypassing internal resampling and eliminating Moiré artifacts.

### 2.2. Global Macro-Shape Guidance
To prevent local patches from losing global context, we first generate a base global depth map. If the input image exceeds four times the patch size, we employ Hierarchical Tiling—recursively generating intermediate reference maps to preserve structural integrity. This global map serves as the low-frequency structural anchor for our fusion algorithm.

### 2.3. Laplacian PatchFusion and Pillowy Distortion Correction
A well-documented artifact of scale-invariant depth models is their tendency to predict local patches as "pillowy" or spherical shapes, centering the depth around the object within the limited receptive field. Naive linear blending preserves these pillowy distortions, resulting in visible grid seams.

To counteract this, we utilize Laplacian PatchFusion. We separate each patch into micro-details (high frequencies) and macro-shape (low frequencies) using a Gaussian blur. Crucially, the choice of the `blur_radius` is paramount. A massive blur radius (e.g., half the patch size) mistakenly categorizes the hallucinated pillowy shape as a micro-detail, injecting it into the final map and causing objects to falsely dim at the edges. By mathematically optimizing the `blur_radius` to isolate true micro-details (e.g., 63 pixels), the pillowy shape is successfully caught by the low-pass filter and replaced by the flat, correct geometry from the global depth map. 

The aligned patch is computed as:
`aligned_local = s * (local_depth - local_blurred) + global_patch_blurred`
where `s` is the least-squares scaling factor derived to match the local and global contrasts.

### 2.4. Cosine Blending and Outer Boundary Preservation
Patches are stitched using a 2D cosine falloff window to ensure the Partition of Unity. However, applying a fade-out at the absolute outer borders of the image results in a zero-weight boundary. When subjected to subsequent frequency separation or blur, this 1-pixel black border bleeds inward (e.g., 32 pixels for a radius 63 blur), creating a dark frame around the entire photograph. We solve this by dynamically cropping the cosine ramp at the absolute image boundaries, clamping the outer weights to `1.0`.

## 3. Optimal Normal Map Generation

Once a seamless, high-resolution depth map is assembled, extracting a mathematically accurate normal map presents unique challenges. Depth networks output disparity (inverse depth) rather than linear depth.

### 3.1. Non-Linear Depth Squaring for Linear Heightfields
Converting disparity directly into 3D space via perspective projection can yield excessive contrast and geometric tearing at object boundaries. To resolve this, we adopt an orthographic displacement approach. However, because objects appear flatter as they recede into the distance, applying standard linear displacement to a disparity map is physically inaccurate. 

We introduce a non-linear mapping by squaring the depth map (`depth_f ** 2`). This mathematical transformation acts as an ideal geometric correction, mapping relative disparities into a linear heightfield $H(x, y)$ that perfectly scales distant objects to appear proportionally flatter, matching human perceptual expectations and perspective foreshortening.

### 3.2. Sobel Gradients and Vector Normalization
The heightfield is scaled by a user-defined bump scale. We compute the spatial gradients using a 3x3 Sobel operator, dividing by 8.0 to normalize the Sobel kernel weight sum:
$$ \nabla_x = \text{Sobel}_x(H \times \text{scale}) / 8.0 $$
$$ \nabla_y = \text{Sobel}_y(H \times \text{scale}) / 8.0 $$
The resulting orthographic normal vector is $N = [-\nabla_x, \nabla_y, 1]$, where the Y-component is inverted to comply with OpenGL normal map standards (Y+ points UP).

### 3.3. Asymmetric Spatial Median Filtering
Despite the Moiré elimination, neural depth maps can still exhibit hallucinated horizontal or vertical stripes (typically ~3 pixels wide) and stray noise. Blurring the normal map would destroy sharp geometric edges. 

To surgically remove these artifacts, we apply a spatial Median Filter independently across all normal channels (X, Y, Z). By utilizing an asymmetric window size of `(8, 5)` (8 pixels vertically, 5 pixels horizontally), the filter mathematically eliminates directional anomalies up to 3-4 pixels thick while perfectly preserving the sharp structural boundaries of the original image geometry.

### 3.4. Hybrid Auto-Calibration via Foreground Surface Tilt Matching
A fundamental limitation of generating normal maps from depth maps is the arbitrary nature of the displacement scale (`bump_scale`). The scale of relative depth varies wildly between different photographs (e.g., a macro shot vs. a landscape), meaning a static `bump_scale` will produce normals that are either far too flat or excessively distorted.

To solve this, we introduce a **Hybrid Auto-Calibration Algorithm**. We utilize the Marigold depth-to-normals diffusion model (which possesses excellent physical volume estimation but suffers from low resolution and pixelation noise) strictly as a structural reference.

1. **Volume Extraction:** We generate a low-resolution Marigold normal map and blur both it and our depth map to discard high-frequency noise, focusing solely on the macro-volumes.
2. **Foreground Isolation:** We isolate the foreground object using a dynamic threshold on the depth map, effectively masking out the background which would otherwise skew mathematical comparisons.
3. **Tilt Target:** We calculate the average magnitude of the X and Y normal components (the "Average Surface Tilt" or bumpiness) within the Marigold foreground.
4. **Logarithmic Optimization Loop:** We iteratively test 40 logarithmic candidate `bump_scale` values (from 50 to 5000) on our depth map. 
5. **Magnitude Matching:** Instead of using Cosine Similarity—which severely penalizes the inevitably misaligned edges between the two models and forces the optimizer toward flat normals—we simply optimize the depth map's scale to exactly match the target Average Surface Tilt.

By doing this, the algorithm autonomously calibrates the mathematically optimal `bump_scale` for any given photograph in milliseconds, yielding a final high-resolution normal map that perfectly inherits the physical volume of Marigold, while maintaining the crisp, noise-free details of DepthAnything V2.

## 4. Conclusion
The proposed Hierarchical Laplacian PatchFusion pipeline, combined with non-linear heightfield squaring and spatial median filtering, provides a robust, artifact-free method for generating ultra-high-resolution depth and normal maps. By carefully navigating the frequency domains of neural network outputs and correcting inherent mathematical distortions at the boundaries, we achieve photorealistic, structurally sound 3D representations from 2D monocular images.
