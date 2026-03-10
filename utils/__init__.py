import torch
import pickle
import os
import torchvision
from torchvision.transforms import Resize

def compute_camera_K(verts, cam):
    B = verts.shape[0]
    K = torch.zeros(B, 4, 4)
    K[:, 0, 0] = -cam[:, 0] 
    K[:, 0, 3] = -cam[:, 0] * cam[:, 1] 
    K[:, 1, 1] = -cam[:, 0] 
    K[:, 1, 3] = -cam[:, 0] * cam[:, 2] 
    K[:, 2, 2] = 1
    K[:, 3, 3] = 1

    return K

def load_meshes(paths):
    verts = torch.tensor([])
    cameras = torch.tensor([])
    for path in paths:
        with open(path, 'rb') as f:
            mesh = pickle.load(f)
            verts = torch.cat((verts, mesh['vertices']), dim=0)
            cameras = torch.cat((cameras, mesh['cam']), dim=0)
            
    return verts, cameras, path

def load_data(frames_path, meshes_path):
    meshes = os.listdir(meshes_path)
    meshes.sort()

    data = []

    for mesh in meshes:
        mesh_name = mesh[:-4]
        frame = os.path.join(frames_path, mesh_name + '.png')
        mesh_path = os.path.join(meshes_path, mesh)

        frame = torchvision.io.read_image(frame)
        frame = Resize((256, 256))(frame)
        
        with open(mesh_path, 'rb') as f:
            mesh = pickle.load(f)
            verts = mesh['vertices']
            cameras = mesh['cam']

        data.append((frame, (verts, cameras), mesh_name))

    return data