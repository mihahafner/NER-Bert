import torch
device = torch.device("cuda")
n_gpu = torch.cuda.device_count()
print(n_gpu)
print(torch.cuda.get_device_name(0))