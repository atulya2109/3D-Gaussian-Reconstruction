# Project Interview Guide — 3D Human Body Avatar from Monocular Video

---

## "Tell me about your project"

The project is an end-to-end pipeline that takes a monocular video of a person and produces an animatable 3D human body avatar using 3D Gaussian Splatting. The pipeline has three main stages:

1. **Segmentation** — SAM2 (Meta's Segment Anything Model v2) isolates the person from the background on every frame, producing per-frame masks.

2. **Body estimation** — SMPLer (a transformer-based body estimator with an HRNet backbone) fits a SMPL parametric body mesh to each frame, producing per-frame 3D vertices and a weak-perspective camera estimate.

3. **Gaussian Splatting** — A GaussianModel is initialised with one 3D Gaussian per triangle of the SMPL UV mesh. Each frame, the Gaussian positions are updated to the triangle centroids of the posed SMPL mesh. The model is trained end-to-end with Huber loss and LPIPS perceptual loss to learn appearance (color, opacity, scale, rotation) while the body motion is driven purely by the SMPL mesh.

The idea is borrowed from FlashAvatar (CVPR 2024) which does the same for head-only avatars using FLAME. This project extends the concept to the full body using SMPL.

---

## "What was the hardest technical challenge?"

### The Camera System — Weak-Perspective vs Full Perspective

**The Problem**

SMPLer runs on monocular video. A monocular camera cannot recover absolute depth — a tall person far away and a short person close up produce identical images. There is no way to separate scale from depth from a single camera. So SMPLer outputs a **weak-perspective camera** `[s, tx, ty]` instead of a full perspective camera:

```
u_ndc = s * X  +  tx
v_ndc = s * Y  +  ty
```

This collapses the focal length `f` and depth `Z` into one number `s = f / Z`. The individual values of `f` and `Z` are unknowable from monocular video alone.

The Gaussian Splatting rasterizer, however, needs a full perspective camera:
- A **view matrix** (world-to-camera transform encoding position and orientation)
- A **projection matrix** (encoding focal length, FOV, near/far planes)
- Real 3D coordinates with actual depth

**Why FlashAvatar Doesn't Have This Problem**

FlashAvatar uses the metrical-tracker / MICA pipeline (Metrical Image-based to Canonical face Avatar). MICA recovers metric 3D face shape from images using a learned statistical prior trained on thousands of 3D face scans. Because the geometry is metric (real-world scale), the tracker outputs a proper OpenCV intrinsic matrix `K` with real focal lengths and proper `R, T` extrinsics — exactly what the rasterizer needs, no conversion required.

This works for faces because face shape is tightly constrained — there is a strong learned prior that allows metric depth recovery from shape alone. The same trick does not extend cleanly to full-body reconstruction, which has far more shape and pose variation.

**The Three Hacks That Resulted**

Working around this with PyTorch3D's `FoVPerspectiveCameras` produced three separate patches:

| Hack | Location | Why it existed |
|------|----------|----------------|
| Negated `s` in K matrix (`-cam[:, 0]`) | `utils/__init__.py` | Trying to stuff weak-perspective scale into a perspective intrinsic matrix |
| Hardcoded `+ [0.5, 1, 20]` vertex offset | `gaussian_model.py` | SMPL vertices sit at Z≈0; pushed manually in front of the camera |
| `proj_mat[:2,:] *= -1` axis flip | `gaussian_renderer/__init__.py` | PyTorch3D NDC flips the X axis vs. the rasterizer's OpenGL convention |

Additionally, the camera was constructed once from frame 0 and reused for all frames — but the weak-perspective scale `s` changes per frame as the person moves.

**The Correct Fix**

Drop PyTorch3D's camera abstraction entirely. Construct the view and projection matrices directly, the same way FlashAvatar does using `getWorld2View2` and `getProjectionMatrix`.

The math to convert `[s, tx, ty]` to a proper 3D camera:

1. **Choose a fixed focal length** `f` in pixels. SMPLer internally assumes `f = 5000` for 224×224 images, which is the correct value to use for consistency with what was assumed during fitting.

2. **Derive depth per frame**: `Z = f / (s * img_size/2)`. This gives the depth at which the SMPL body must sit so that the perspective projection matches the weak-perspective estimate.

3. **Derive 3D translation**: `T = [tx/s, ty/s, Z]`. This positions the body center so it projects to `(tx, ty)` in normalised image space.

4. **Build the view matrix**: `getWorld2View2(R=identity, T)`.

5. **Build the projection matrix**: `getProjectionMatrix(znear, zfar, fovX, fovY)` where `fovX = 2 * atan(img_size / (2*f))`.

This removes all three hacks and correctly propagates per-frame camera changes.

**If COLMAP Intrinsics Are Available**

If the video was processed through COLMAP (structure from motion), the `sparse/0/cameras.txt` file contains the real focal length of the recording camera. Using that `f` directly eliminates the last degree of freedom in the conversion and makes the reconstruction fully metric — equivalent to what FlashAvatar gets from MICA.

---

## "What rendering artifacts did you observe and what caused them?"

### Artifact 1 — Runaway Gaussians (Large Oval Blob)

**What it looked like**: A large dark oval shape appearing on top of the rendered person, sometimes filling most of the image.

**Root cause**: Missing activation functions on Gaussian parameters before passing them to the rasterizer.

The `diff-gaussian-rasterization` CUDA kernel expects:
- **Scales** — positive values (optimizer should work in log space, apply `exp` before rasterizer)
- **Rotations** — unit quaternions (must be L2-normalised)
- **Opacity** — values in `[0, 1]` (apply `sigmoid` before rasterizer)

The original 3DGS code always activates these:
```python
scales    = torch.exp(self._scaling)
rotations = F.normalize(self._rotation, dim=-1)
opacity   = torch.sigmoid(self._opacity)
```

In this project they were passed raw:
```python
scales    = pc.scaling    # no exp
rotations = pc.rotation   # no normalisation
opacity   = pc.opacity    # no sigmoid
```

When quaternions are not normalised, the 3D covariance matrix `Σ = R * S * S^T * R^T` is computed from an invalid rotation matrix. A quaternion that drifts slightly off-norm compounds through the covariance computation into a massively elongated ellipse in screen space — visible as the large oval. Unconstrained scale allows a single Gaussian to balloon to fill the entire image.

**Fix**: Add properties with the activations applied and use those in the renderer instead of the raw parameters.

---

### Artifact 2 — Blurry Renders

**What it looked like**: The person was recognisable and roughly correct in shape and color, but all fine detail — face, hands, clothing texture — was soft and unresolved.

**Root cause**: No adaptive densification.

The original 3DGS paper includes an adaptive densification loop that runs every N iterations:
- **Clone** — if a Gaussian has high positional gradient but small scale (it is in an under-reconstructed region), spawn a copy nearby
- **Split** — if a Gaussian is too large and trying to cover complex geometry, replace it with two smaller Gaussians
- **Prune** — if a Gaussian has near-zero opacity (it contributes nothing), remove it

This project initialises exactly one Gaussian per SMPL mesh triangle and never changes that count. Every Gaussian is responsible for a fixed area of the body surface. Regions that need more detail — face, hands, fabric folds — cannot receive more Gaussians. The model compensates by spreading each Gaussian wide, which produces blur.

The renderer already returns the signal needed for densification — `viewspace_points` gradients and `radii` — but they are never used.

**Fix**: Port the densification logic from FlashAvatar's `scene/gaussian_model.py`, which already implements `densify_and_clone`, `densify_and_split`, and `prune_points`. The key inputs are the `viewspace_points` gradient magnitude (to detect under-reconstructed regions) and Gaussian scale (to detect over-large Gaussians).

**Secondary contributing factor**: A single fixed training viewpoint. Because training only ever sees the person from one camera angle, Gaussians only need to be consistent from that direction. SH coefficients with degree 3 (16 bands) are also more than needed for single-viewpoint training and slow convergence.

---

## "How do you control Gaussian positions after densification?"

This is a fundamental architectural question. The current approach places one Gaussian at the centroid of each SMPL mesh triangle:

```python
triangles = verts[self.faces.verts_idx]
self.xyz = triangles.mean(dim=1)   # one centroid per face
```

This relies on a strict 1-to-1 mapping between Gaussians and faces. Densification breaks that — after splitting and cloning, you have more Gaussians than faces and the direct lookup no longer works.

### The Fix: Barycentric Coordinates

Instead of storing a face index and always placing the Gaussian at the centroid, each Gaussian stores:
- `face_idx` — which SMPL triangle it belongs to
- `bary` — barycentric coordinates `(α, β, γ)` within that triangle

The 3D position each frame becomes:

```python
tri_per_gaussian = triangles[self.face_idx]   # look up this Gaussian's triangle
xyz = (self.bary.unsqueeze(-1) * tri_per_gaussian).sum(dim=1)
```

The centroid approach is just a special case of this where every Gaussian has `bary = (1/3, 1/3, 1/3)`. When densification creates a new Gaussian via clone, it inherits the parent's `face_idx` and `bary`. When splitting, it inherits the parent's `face_idx` but perturbs `bary` slightly in opposite directions. Every Gaussian always has a mesh attachment, regardless of how many are created.

A normal offset per Gaussian can also be added — a learnable scalar that displaces the Gaussian along the face normal — allowing Gaussians to float slightly off the mesh surface for hair, loose clothing, and details that the SMPL geometry doesn't capture.

---

## "Have you considered alternative ways to initialise and control Gaussians on a body mesh?"

### The UV Map Approach

An alternative raised in the literature is to initialise Gaussians not from triangle centroids but from a **regular grid in UV texture space**.

A UV map is a 2D parameterisation of the 3D mesh surface — every point on the body gets a coordinate `(u, v)` in `[0, 1]²`. The SMPL mesh ships with a UV map (`smpl_uv.obj`), which is already loaded in this project's `GaussianModel`.

Instead of one Gaussian per triangle, you define a grid — say 256×256 — in UV space, giving 65,536 Gaussians uniformly distributed across the body surface. For each grid point you precompute once (static, pose-independent):
- Which SMPL triangle contains this `(u, v)` coordinate
- The barycentric coordinates of `(u, v)` within that triangle

Then each frame, position is recovered identically to the barycentric approach above:

```python
# precomputed once
# gaussian_face_idx: [N_gaussians]    which triangle each UV point maps to
# gaussian_bary:     [N_gaussians, 3] barycentric coords within that triangle

tri_per_gaussian = triangles[gaussian_face_idx]
xyz = (gaussian_bary.unsqueeze(-1) * tri_per_gaussian).sum(dim=1)
```

**Why this is better than triangle centroids:**

| | Triangle centroids (current) | UV grid |
|---|---|---|
| Gaussian count | Fixed at one per face (~13k for SMPL) | Any resolution — 64², 256², 512² |
| Distribution | Uneven — follows triangle size variation | Uniform across body surface |
| Density control | None without densification | Change grid resolution |
| Densification needed | Yes, to add detail | No — just increase grid resolution |
| Mesh binding after densification | Breaks the 1-to-1 mapping | Never an issue — all positions are UV-derived |

**Why SMPL triangle density is uneven**: The face and hands have many small triangles (high mesh detail), while the torso has large triangles. Triangle centroids therefore cluster Gaussians in regions SMPL already models well, and under-sample regions that would benefit from more Gaussians for appearance.

A 256×256 UV grid gives roughly uniform coverage regardless of mesh triangle size, and you can trivially increase it to 512×512 for more detail.

**The tradeoff**: UV parameterisations have seams — discontinuities where the mesh was cut to unfold flat. Gaussians near a seam in UV space are not neighbours in 3D space. For learning smooth appearance features this can cause visible artifacts at seam boundaries, which need to be handled explicitly (e.g. by padding the UV map at seam edges or masking seam regions during training).

---

## "What would you do differently or what are the next steps?"

### 1. Fix the Activation Functions (Quick Win)
Add `torch.exp`, `F.normalize`, and `torch.sigmoid` as properties on `GaussianModel`. This immediately eliminates the runaway Gaussian artifact with no architectural changes.

### 2. Implement Densification (Biggest Quality Improvement)
Port densification from FlashAvatar or the original 3DGS codebase. This is the single change most likely to improve render quality from blurry to sharp. Requires migrating Gaussian positions to the barycentric coordinate system first so mesh control is preserved after densification.

### 3. Migrate to Barycentric or UV-Based Gaussian Control
Replace the triangle centroid approach with either barycentric coordinates (for densification compatibility) or a UV grid (for uniform coverage and resolution control without densification).

### 4. Proper Camera Conversion (Correctness)
Replace the PyTorch3D camera hacks with direct matrix construction using `getWorld2View2` and `getProjectionMatrix`. Use `focal = 5000` for consistency with SMPLer's internal assumption, or the real COLMAP focal length if available.

### 5. Use Per-Frame Camera (Correctness)
The training loop currently builds the camera once from frame 0. Each frame's `cam [s, tx, ty]` should be used to build a per-frame view matrix so the projection is correct for every frame, not just the first.

### 6. Upstream Estimator Upgrade (Metric Depth)
Replacing SMPLer with CLIFF (Carries Location Information in Full Frames) or 4DHumans would provide better depth estimates. These methods use the full image frame rather than a cropped region, which gives them the scale context to estimate focal length and recover more accurate absolute depth — partially solving the weak-perspective problem at the source.

---

## "How does your approach compare to FlashAvatar?"

| | FlashAvatar | This Project |
|---|---|---|
| Body part | Head only (FLAME) | Full body (SMPL) |
| Tracker | metrical-tracker / MICA | SMPLer |
| Camera params | Real metric K, R, T from tracker | Weak-perspective [s, tx, ty] |
| Depth recovery | Metric (from MICA's 3D prior) | Approximated from scale `s` |
| Gaussians | Anchored to FLAME mesh + deformation MLP | Anchored to SMPL mesh, no deformation |
| Densification | Yes | Not yet implemented |
| Render quality | High fidelity, 300+ FPS | Functional but blurry without densification |

The core architectural idea is identical. The differences are: the body model scope (head vs. full body), the depth ambiguity introduced by monocular full-body estimation, and the missing densification which limits current render quality.

---

## Quick Reference — Key Concepts

**SMPL** — Skinned Multi-Person Linear model. A parametric 3D body model with shape parameters `β` (body shape) and pose parameters `θ` (joint rotations). Outputs 6890 mesh vertices.

**3D Gaussian Splatting** — A scene representation using a set of 3D Gaussians, each with position, rotation, scale, opacity, and spherical harmonic color coefficients. Rendered via differentiable rasterization at real-time speeds.

**Weak-perspective camera** — An approximation of perspective projection that assumes all points in a scene are at the same depth. Encodes focal length and depth as a single scale factor `s = f/Z`. Used when absolute depth cannot be recovered.

**LPIPS** — Learned Perceptual Image Patch Similarity. A perceptual loss that measures image similarity in the feature space of a pretrained VGG network rather than pixel space. Better at driving sharp, visually realistic outputs than L2 or Huber loss alone.

**Densification** — The adaptive process in 3DGS of splitting, cloning, and pruning Gaussians during training based on gradient signals. Essential for achieving fine detail in the final render.

**COLMAP** — Structure from Motion pipeline that recovers camera intrinsics (focal length, principal point) and extrinsics (position, orientation) from a set of images by matching features across frames.

**UV map** — A 2D parameterisation of a 3D mesh surface. Every point on the mesh is assigned a coordinate `(u, v)` in `[0, 1]²`, effectively "unfolding" the surface flat. Used in this context to define a uniform grid of Gaussian positions on the body surface without being tied to mesh triangle topology.

**Barycentric coordinates** — A way to express a point's position within a triangle as a weighted combination of the three vertices: `P = α*v0 + β*v1 + γ*v2` where `α + β + γ = 1`. Used to bind Gaussians to specific locations on the mesh surface so they deform correctly with pose changes, regardless of how many Gaussians exist per triangle.

**Linear Blend Skinning (LBS)** — The deformation method used by SMPL. Each vertex is assigned weights across skeleton joints. When the skeleton pose changes, each vertex moves as a weighted average of the joint transformations. Barycentric-bound Gaussians inherit this deformation implicitly through their triangle vertices.
