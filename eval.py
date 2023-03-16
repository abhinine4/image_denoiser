import os
import io
import numpy as np
import PIL.Image as pil
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
from dncnn import Dncnn
from config import *

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if arch == 'Dncnn-S':
        model = Dncnn(num_layers=17)
    elif arch == 'Dncnn-B':
        model = Dncnn(num_layers=20)
    elif arch == 'Dncnn-3':
        model = Dncnn(num_layers=20)

    state_dict  = model.state_dict()
    for n,p in torch.load(weights_path, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)
    
    model = model.cuda()
    model.eval()

    #image_path

    filename = os.path.basename(image_path).split('.')[0]
    descriptions = ''

    input = pil.open(image_path).convert('RGB')

    if gnl is not None:
        gnl = int(gnl)
        noise = np.random.normal(0.0, gnl, (input.height, input.width, 3)).astype(np.float32)
        input = np.array(input).astype(np.float32)+noise
        descriptions += '_noise_{}'.format(gnl)
        pil.fromarray(input.clip(0.0, 255.0).astype(np.uint8)).save(os.path.join(output_dir, '{}{}.png'.format(filename, descriptions)))
        input /= 255.0

    if jq is not None:
        buffer = io.BytesIO()
        input.save(buffer, format='jpeg', quality=jq)
        input = pil.open(buffer)
        descriptions += '_jpeg_q{}'.format(jq)
        input.save(os.path.join(output_dir, '{}{}.png'.format(filename, descriptions)))
        input = np.array(input).astype(np.float32)
        input /= 255.0

    if df is not None:
        original_width = input.width
        original_height = input.height
        input = input.resize((input.width // df,
                              input.height // df),
                             resample=pil.BICUBIC)
        input = input.resize((original_width, original_height), resample=pil.BICUBIC)
        descriptions += '_sr_s{}'.format(df)
        input.save(os.path.join(output_dir, '{}{}.png'.format(filename, descriptions)))
        input = np.array(input).astype(np.float32)
        input /= 255.0
    
    input = transforms.ToTensor()(input).unsqueeze(0).cuda()

    with torch.no_grad():
        pred = model(input)
    
    output = pred.mul_(255.0).clamp_(0.0, 255.0).squeeze(0).permute(1,2,0).byte().cpu().numpy()
    output = pil.fromarray(output, mode='RGB')
    output.save(os.path.join(output_dir, '{}{}_{}_#{}.png'.format(filename, descriptions, arch, num_epochs)))
