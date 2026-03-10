import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from models import GaussianModel

def render(viewpoint_camera, pc : GaussianModel, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.xyz, dtype=pc.xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = torch.tan(torch.deg2rad(viewpoint_camera.fov) * 0.5)
    tanfovy = torch.tan(torch.deg2rad(viewpoint_camera.fov) * 0.5)

    proj_mat = viewpoint_camera.get_full_projection_transform().get_matrix()
    view_matrix = viewpoint_camera.get_world_to_view_transform().get_matrix().squeeze(0)
    proj_mat = proj_mat.squeeze(0)
    proj_mat[:2,:] *= -1

    raster_settings = GaussianRasterizationSettings(
        image_height=int(256),
        image_width=int(256),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=view_matrix,
        projmatrix=proj_mat,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.get_camera_center().squeeze(0),
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    means3D = pc.xyz
    means2D = screenspace_points
    opacity = pc.opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    
    scales = pc.scaling
    rotations = pc.rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        shs = pc.features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D, # pass torch.zeros_like(means3D) to get gradients of 2D means
        shs = shs,
        colors_precomp = colors_precomp, # None
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp) # None

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
