import torch

model_path = '/train33/sppro/permanent/hangchen2/pandora/egs/misp2024-mmmt-minggao/examples/misp2022/s0/exp_vsr/vsr/39.pt'
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))


#for p,pt in checkpoint['model_state_dict'].items():
for p,pt in checkpoint.items():
    #if p.startswith('trunk') or p.startswith('frontend3D'):
    print(p, pt.shape)