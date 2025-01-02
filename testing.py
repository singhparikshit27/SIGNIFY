import torch
print("Is CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Current device:", torch.cuda.get_device_name(torch.cuda.current_device()))
