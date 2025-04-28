import numpy as np
import torch, torchvision
from pathlib import Path
from tqdm import tqdm


contact_dir = Path('data/contact_maps')  
out_dir     = Path('data/resnet_data')
out_dir.mkdir(parents=True, exist_ok=True)

model = torchvision.models.resnet50(weights='IMAGENET1K_V2')
model.fc = torch.nn.Identity()            
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.eval().to(device)

for npy_path in tqdm(list(contact_dir.glob('*.npy')), desc='ResNet embedding'):
    entry = npy_path.stem                 
    out_pt = out_dir / f'{entry}.pt'
    if out_pt.exists():                   
        continue

    contact_map = np.load(npy_path)             
    if contact_map.ndim == 3:
        contact_map = contact_map[:, :, 0]


    cmap3  = np.stack([contact_map]*3, axis=0)
    tensor = torch.tensor(cmap3, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(tensor).cpu().squeeze(0)   
    torch.save(emb, out_pt)

