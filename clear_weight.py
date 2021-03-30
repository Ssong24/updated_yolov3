from utils.utils import *

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')
    mixed_precision = False  # not installed

device, _ = torch_utils.select_device('0', apex=mixed_precision, batch_size=20)
ckpt = torch.load('input/pretrained_weights/yolov3-spp-concatv3_nofix.pt', map_location=device)

# Display the ckpt keys (ckpt is dictionary data type)
# print(ckpt.keys())
# My pretrained model has data inside ckpt['optimizer'], ckpt['training_results'], ckpt['epoch']
# For transfer learning, I need to remove those above information in ckpt
ckpt['optimizer'] = None
ckpt['training_results'] = None
ckpt['epoch'] = -1

# Save the checkpoint
torch.save(ckpt, 'input/pretrained_weights/yolov3-spp-concatv3.pt')
