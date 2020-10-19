# @jkarki
# I'm trying to check if changing the dataloaders will prevent us from running out of memory

import dataloaders
import torch
import numpy as np
from tqdm import tqdm
import argparse
import os
import torch.nn.functional as F
import time
from ops.utils_blocks import block_module
from ops.utils import show_mem, generate_key, save_checkpoint, str2bool, step_lr, get_lr
torch.Tensor.__str__ = lambda x: str(tuple(x.shape))

# @jkarki, importing the new method 
from ops.utils import mlr_generate_key

parser = argparse.ArgumentParser()
#model
parser.add_argument("--mode", type=str, default='group',help='[group, sc]')
parser.add_argument("--stride", type=int, dest="stride", help="stride size", default=1)
parser.add_argument("--num_filters", type=int, dest="num_filters", help="Number of filters", default=256)
parser.add_argument("--kernel_size", type=int, dest="kernel_size", help="The size of the kernel", default=7)
parser.add_argument("--noise_level", type=int, dest="noise_level", help="Should be an int in the range [0,255]", default=25)
parser.add_argument("--unfoldings", type=int, dest="unfoldings", help="Number of LISTA step unfolded", default=24)
parser.add_argument("--patch_size", type=int, dest="patch_size", help="Size of image blocks to process", default=56)
parser.add_argument("--rescaling_init_val", type=float, default=1.0)
parser.add_argument("--lmbda_prox", type=float, default=0.02, help='intial threshold value of lista')
parser.add_argument("--spams_init", type=str2bool, default=0, help='init dict with spams dict')
parser.add_argument("--nu_init", type=float, default=1, help='convex combination of correlation map init value')
parser.add_argument("--corr_update", type=int, default=3, help='choose update method in [2,3] without or with patch averaging')
parser.add_argument("--multi_theta", type=str2bool, default=1, help='wether to use a sequence of lambda [1] or a single vector during lista [0]')
parser.add_argument("--diag_rescale_gamma", type=str2bool, default=0,help='diag rescaling code correlation map')
parser.add_argument("--diag_rescale_patch", type=str2bool, default=1,help='diag rescaling patch correlation map')
parser.add_argument("--freq_corr_update", type=int, default=6, help='freq update correlation_map')
parser.add_argument("--mask_windows", type=int, default=1,help='binarym, quadratic mask [1,2]')
parser.add_argument("--center_windows", type=str2bool, default=1, help='compute correlation with neighboors only within a block')
parser.add_argument("--multi_std", type=str2bool, default=0)

#training
parser.add_argument("--lr", type=float, dest="lr", help="ADAM Learning rate", default=6e-4)
parser.add_argument("--lr_step", type=int, dest="lr_step", help="ADAM Learning rate step for decay", default=80)
parser.add_argument("--lr_decay", type=float, dest="lr_decay", help="ADAM Learning rate decay (on step)", default=0.35)
parser.add_argument("--backtrack_decay", type=float, help='decay when backtracking',default=0.8)
parser.add_argument("--eps", type=float, dest="eps", help="ADAM epsilon parameter", default=1e-3)
parser.add_argument("--validation_every", type=int, default=10, help='validation frequency on training set (if using backtracking)')
parser.add_argument("--backtrack", type=str2bool, default=1, help='use backtrack to prevent model divergence')
parser.add_argument("--num_epochs", type=int, dest="num_epochs", help="Total number of epochs to train", default=300)
parser.add_argument("--train_batch", type=int, default=32, help='batch size during training')
parser.add_argument("--test_batch", type=int, default=10, help='batch size during eval')
parser.add_argument("--aug_scale", type=int, default=0)

#save
# @jkarki, making the default out_dir from trained_model to trained_model_2
parser.add_argument("--out_dir", type=str, dest="out_dir", help="Results' dir path", default='./trained_model2')
parser.add_argument("--model_name", type=str, dest="model_name", help="The name of the model to be saved.", default=None)
parser.add_argument("--resume", type=str2bool, dest="resume", help='Resume training of the model',default=True)
parser.add_argument("--dummy", type=str2bool, dest="dummy", default=False)
parser.add_argument("--tqdm", type=str2bool, default=False)
parser.add_argument("--test_path", type=str, help="Path to the dir containing the testing datasets.", default="./datasets/BSD68/")
parser.add_argument("--train_path", type=str, help="Path to the dir containing the training datasets.", default="./datasets/BSD400/")

#inference
parser.add_argument("--stride_test", type=int, default=10, help='stride of overlapping image blocks [4,8,16,24,48] kernel_//stride')
parser.add_argument("--stride_val", type=int, default=50, help='stride of overlapping image blocks for validation [4,8,16,24,48] kernel_//stride')
parser.add_argument("--test_every", type=int, default=100, help='report performance on test set every X epochs')
parser.add_argument("--block_inference", type=str2bool, default=True,help='if true process blocks of large image in paralel')
parser.add_argument("--pad_image", type=str2bool, default=0,help='padding strategy for inference')
parser.add_argument("--pad_block", type=str2bool, default=1,help='padding strategy for inference')
parser.add_argument("--pad_patch", type=str2bool, default=0,help='padding strategy for inference')
parser.add_argument("--no_pad", type=str2bool, default=False, help='padding strategy for inference')
parser.add_argument("--custom_pad", type=int, default=None,help='padding strategy for inference')

#var reg
parser.add_argument("--nu_var", type=float, default=0.01)
parser.add_argument("--freq_var", type=int, default=3)
parser.add_argument("--var_reg", type=str2bool, default=False)

# @jkarki, idk what the argument means but it's required for the csv report writing thing at the end
parser.add_argument("--tb", type=str2bool, default=False)
####

# @jkarki, add a new argument to specify the GPU device id
parser.add_argument("--device_id", type=int, default=0, help="the device id (0-3) for tacc nodes to select a particular gpu to train on") 
####

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# @jkarki, set the torch device into the one from args
torch.cuda.set_device(args.device_id)
####

device_name = torch.cuda.get_device_name(args.device_id) if torch.cuda.is_available() else 'cpu'
capability = torch.cuda.get_device_capability(args.device_id) if torch.cuda.is_available() else os.cpu_count()

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

test_path = [f'{args.test_path}']
train_path = [f'{args.train_path}']
val_path = train_path

noise_std = args.noise_level / 255

loaders = dataloaders.get_dataloaders(train_path, test_path, val_path, crop_size=args.patch_size,
                                      batch_size=args.train_batch, downscale=args.aug_scale, concat=1)


if args.mode == 'group':
    print('group mode')
    from model.color_group import ListaParams
    from model.color_group import groupLista as Lista

    params = ListaParams(kernel_size=args.kernel_size, num_filters=args.num_filters, stride=args.stride,
                         unfoldings=args.unfoldings, freq=args.freq_corr_update,corr_update=args.corr_update,
                         lmbda_init=args.lmbda_prox, h=args.rescaling_init_val,spams=args.spams_init,multi_lmbda=args.multi_theta,
                         center_windows=args.center_windows,std_gamma=args.diag_rescale_gamma,
                         std_y=args.diag_rescale_patch,block_size=args.patch_size,nu_init=args.nu_init,mask=args.mask_windows, multi_std=args.multi_std,
                         freq_var=args.freq_var, var_reg=args.var_reg, nu_var=args.nu_var)

elif args.mode == 'sc':
    print('sc mode')
    from model.color_sc import ListaParams
    from model.color_sc import Lista

    params = ListaParams(kernel_size=args.kernel_size, num_filters=args.num_filters, stride=args.stride,
                         unfoldings=args.unfoldings,threshold=args.lmbda_prox, multi_lmbda=args.multi_theta)
else:
    raise NotImplementedError

model = Lista(params).to(device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=args.eps)

if args.backtrack:
    reload_counter = 0

pytorch_total_params = sum(p.numel() for p in model.parameters())
print(f'Arguments: {vars(args)}')
print('Nb tensors: ',len(list(model.named_parameters())), "; Trainable Params: ", pytorch_total_params, "; device: ", device,
      "; name : ", device_name)

psnr = {x: np.zeros(args.num_epochs) for x in ['train', 'test', 'val']}


#### NEW CODE, @jonykarki and @asimsedhain
# table_lookup: CD_METHOD_NOISE-LEVEL_BATCH-SIZE
# CD: Color Denoising, Method: SC or GroupSC, the noise_level, and training batch size.
table_lookup = f"CD_{args.mode}_{args.noise_level}_{args.train_batch}"
model_name = args.model_name if args.model_name is not None else mlr_generate_key(table_lookup)

# model_name = args.model_name if args.model_name is not None else generate_key()
####
out_dir = os.path.join(args.out_dir, model_name)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

ckpt_path = os.path.join(out_dir+'/ckpt')
config_dict = vars(args)

if args.resume:
    if os.path.isfile(ckpt_path):
        try:
            print('\n existing ckpt detected')
            checkpoint = torch.load(ckpt_path)
            start_epoch = checkpoint['epoch']
            psnr_validation = checkpoint['psnr_validation']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{ckpt_path}' (epoch {start_epoch})")
        except Exception as e:
            print(e)
            print(f'ckpt loading failed @{ckpt_path}, exit ...')
            exit()

    else:
        print(f'\nno ckpt found @{ckpt_path}')
        start_epoch = 0
        psnr_validation = 22.0
        if args.backtrack:
            state = {'psnr_validation': psnr_validation,
                     'epoch': 0,
                     'config': config_dict,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict(), }
            torch.save(state, ckpt_path + '_lasteval')

print(f'... starting training ...\n')

l = args.kernel_size // 2
mask = F.conv_transpose2d(torch.ones(1, 1, args.patch_size - 2 * l, args.patch_size - 2 * l),
                          torch.ones(1, 1, args.kernel_size, args.kernel_size))
mask /= mask.max()
mask = mask.to(device=device)

epoch = start_epoch


# @jkarki, rewriting the training code with modified dataloader (mlr_dataloader.py)

import mlr_dataloaders
train_dataloader = mlr_dataloaders.get_train_dataloader(train_path, crop_size=args.patch_size, batch_size=args.train_batch, downscale=args.aug_scale)

num_iters = 0
psnr_set = 0

while epoch < args.num_epochs:
	tic = time.time()
	for batch in tqdm(train_dataloader, disable=not args.tqdm):
		batch = batch.to(device=device)
		noise = torch.randn_like(batch) * noise_std
		noisy_batch = batch + noise
		optimizer.zero_grad()

		with torch.set_grad_enabled(True):
			output = model(noisy_batch)
			loss = (mask * (output-batch)).pow(2).sum() / batch.shape[0]
			loss_psnr = -10 * torch.log10((output - batch).pow(2).mean([1,2,3])).mean()
			loss.backward()
			optimizer.step() 	

		psnr_set += loss_psnr.item()
		num_iters += 1

	tac = time.time()
	psnr_set /= num_iters
	
	epoch += 1
	print(psnr_set, epoch)

####


if args.tb and args.model_name is not None:
    import csv
    import re
    epoch-= 1
    score = [f'{psnr[phase][-1]:0.4f}' for phase in ['train','val','test']]
    row = re.findall('.+?%.+?_', args.model_name) + score
    # row = args.model_name.split('_') + score
    with open(f'{args.out_dir}/report.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    csvFile.close()
