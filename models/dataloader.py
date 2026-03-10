import os
import torchvision.transforms as T
from torch.utils.data import Dataset
import torchvision.io
from utils import load_meshes
import pickle

class FrameMeshDataset(Dataset):
    def __init__(self, frames_dir, meshes_dir, device):
        self.frames_dir = frames_dir
        self.meshes_dir = meshes_dir
        self.device = device
        self.frames = sorted(os.listdir(frames_dir))
        self.data = self.load_data()
        
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        return {'frame': self.data[idx][0], 'verts': self.data[idx][1][0], 'cameras': self.data[idx][1][1], 'name': self.data[idx][2]}


    def load_data(self):
        data = []
        for frame in self.frames:
            frame_name = frame[:-4]
            frame_path = os.path.join(self.frames_dir, frame)
            mesh_name = frame_name + '.pkl'
            mesh_path = os.path.join(self.meshes_dir, mesh_name)
            
            frame = torchvision.io.read_image(frame_path).float()/255
            
            with open(mesh_path, 'rb') as f:
                mesh = pickle.load(f)
                verts = mesh['vertices']
                cameras = mesh['cam']
            
            data.append((frame, (verts[0], cameras[0]), frame_name))
        
        return data
