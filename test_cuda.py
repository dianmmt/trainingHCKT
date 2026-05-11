import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    mem = torch.cuda.get_device_properties(0).total_memory/1e9
    print(f'VRAM: {mem:.1f} GB')