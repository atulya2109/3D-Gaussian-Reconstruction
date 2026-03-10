from torch import nn
from pytorch3d.io import load_obj
import torch
from torchvision import transforms as T
import pickle 
from torch.optim import Adam

class GaussianModel(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.active_sh_degree = 3

        _, self.faces, _ = load_obj('./smpl/smpl_uv.obj', load_textures=False)

    def init_gaussians(self):
        faces_idx = self.faces.verts_idx
        n_of_gaussians = faces_idx.shape[0]

        rotation = torch.zeros((n_of_gaussians, 4), device=self.device)

        rotation[:, 0] = 1 # w component of the quaternion
        self.rotation = nn.Parameter(rotation)

        opacity = torch.ones((n_of_gaussians, 1), device=self.device) *0.8
        self.opacity = nn.Parameter(opacity)

        sh = torch.rand((n_of_gaussians, 16, 3), device=self.device)
        self.features = nn.Parameter(sh)

        scale = torch.ones((n_of_gaussians, 3), device=self.device) * 0.00001
        self.scaling = nn.Parameter(scale)

        self.xyz = torch.rand((n_of_gaussians, 3), device=self.device)
        # self.xyz = nn.Parameter(torch.rand((n_of_gaussians, 3), device=self.device))

    def update_xyz(self, verts):
        transform = T.Compose([
            T.Lambda(lambda x: x + torch.tensor([0.5, 1, 20], device=x.device))
        ])
        verts = transform(verts)
        verts = verts.to(torch.float32)
        triangles = verts[self.faces.verts_idx]

        triangle_centers = triangles.mean(dim=1).to(self.device)

        # with torch.no_grad():
        #     self.xyz.data.copy_(triangle_centers)

        self.xyz = triangle_centers

    def get_optimizer(self, lr):
        self.optimizer = Adam([
            {'params': self.rotation, 'lr': lr['rotation']},
            {'params': self.opacity, 'lr': lr['opacity']},
            {'params': self.features, 'lr': lr['features']},
            {'params': self.scaling, 'lr': lr['scaling']},
            # {'params': self.xyz, 'lr': lr['xyz']}
        ])
            
        return self.optimizer
    
    def capture(self):
        return (self.rotation, self.opacity, self.features, self.scaling, self.xyz, self.optimizer.state_dict())
        



        


    
        



        

        