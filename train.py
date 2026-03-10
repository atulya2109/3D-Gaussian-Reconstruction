from models import GaussianModel, FrameMeshDataset
from gaussian_renderer import render
from pytorch3d.renderer import FoVPerspectiveCameras
from utils import compute_camera_K, load_meshes
import torch 
import os
import torchvision.transforms as T
import lpips
from torch.nn import HuberLoss
from tqdm import tqdm
from torchvision.utils import make_grid
import sys
from datetime import datetime

device = "cuda"

gm = GaussianModel(device=device)
gm.init_gaussians()

chkpnt_dir = "data/1/chkpnt"

dataset = FrameMeshDataset(frames_dir='data/1/processed', meshes_dir='data/1/meshes', device='cuda')

dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
bg_color = torch.tensor([1, 1, 1], dtype=torch.float32,device=device)

verts, cameras, _ = load_meshes(['data/1/meshes/1_00000.pkl'])

camera = FoVPerspectiveCameras(device=device, K=compute_camera_K(verts, cameras))


max_iterations = 5000

huber_loss = HuberLoss()
lpips_loss = lpips.LPIPS(net='vgg').to(device)

lr = {
    'rotation': 0.001,
    'scaling': 0.001,
    'opacity': 0.01,
    'features': 0.005,
    # 'xyz': 0.005
}

gm.get_optimizer(lr=lr)
print("\rStarting training...")
p = tqdm(total=max_iterations)

current_time = datetime.now().strftime("%m_%d-%H:%M")
model_path = f"data/1/chkpnt/no_xyz|{current_time}"
visual_path = f"data/1/output/no_xyz|{current_time}"

os.makedirs(visual_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

for i in range(1, max_iterations+1):
    p2 = tqdm(total=len(dataloader))
    for data in dataloader:
        gm.optimizer.zero_grad()
        renders = []
        for frame, verts, cameras , name in zip(data['frame'], data['verts'], data['cameras'], data['name']):

            frame = frame.to(device).float()
            gm.update_xyz(verts)
            output = render(camera, gm, bg_color, scaling_modifier=1.0, override_color=None)
            
            img = output['render']
            
            renders.append(img)

        renders = torch.stack(renders, dim=0)

        loss_huber = huber_loss(renders, data['frame'].to(device)).mean()
        loss_lpips = 0

        if i > 500:
            loss_lpips = lpips_loss(renders, data['frame'].to(device)).mean()

        loss = loss_huber + loss_lpips
        
        loss.backward()
        
        gm.optimizer.step()

        p2.set_description(f"Loss: {loss.item():.4f} LPIPS: {loss_lpips:.4f} Huber: {loss_huber.item():.4f} grad: {gm.features.grad.mean().item()}")
        p2.update(1)

    if i % 100 == 0 or i==1:
        torch.save(gm.capture(), os.path.join(model_path, f"chkpnt_{i}.pt"))
        out = torch.stack([renders.clamp(0.,1.), data['frame'].to(device)], dim=0)
        out = out.transpose(0,1).reshape(-1, *out.shape[2:])
        T.ToPILImage()(make_grid(out)).save(os.path.join(visual_path, f"img_{i}.png"))

    p2.close()
    p.update(1)

p.close()