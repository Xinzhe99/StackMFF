# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
# @Description : Training script for the StackMFF model for multi-focus image fusion.
#                This script trains the model on multi-focus image stacks and their corresponding
#                all-in-focus ground truth images.
#
# @Usage Example:
#   python train.py --datapath path/to/training_datasets --exp_name stackmff_experiment --epochs 20 --batch_size 16
#   python train.py --datapath path/to/training_datasets --exp_name stackmff_experiment --lr 1e-3 --loss_weights 0.2 0.6 0.1 0.1
#   python train.py --datapath path/to/training_datasets --exp_name stackmff_experiment --image_size 512 512 --gamma 0.8

import torch
import torch.nn as nn
import argparse
import time
from tqdm import tqdm
import os
import Dataloader
import loss
from stackmff import StackMFF
from torch.utils.tensorboard import SummaryWriter
import collections

start_full_time = time.time()
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='StackMFF')
parser.add_argument('--exp_name', default='experiment', help='experiment name for saving models')
parser.add_argument('--datapath', default=r'path/to/training_datasets',help='datasets path')
parser.add_argument('--image_size', type=int, nargs=2, default=[384, 384], help='image size for training (height, width)')
parser.add_argument('--epochs', type=int, default=20,help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,help='16 default')
parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
parser.add_argument('--lr', type=float, default=5e-3,help='5e-3 default')
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--model_save_fre', type=int, default=1,help='model save frequence (default: 1)')
parser.add_argument('--loss_weights', type=float, nargs=4, default=[0.1, 0.7, 0.1, 0.1], 
                    help='loss weights for L1, SSIM, Laplacian, and SpatialFrequency losses')
parser.add_argument('--gpu_id', type=int, nargs='+', default=None, help='GPU ID(s) to use. None for all available GPUs, [0] for GPU 0 only.')

args = parser.parse_args()

def get_stack_depth(folder_path):
    """Get the number of images in a stack folder"""
    import os
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                   f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]
    return len(image_paths)

def group_samples_by_depth(img_stack, label_img):
    """Group samples by stack depth"""
    depth_groups = collections.defaultdict(list)
    for i, stack_path in enumerate(img_stack):
        depth = get_stack_depth(stack_path)
        depth_groups[depth].append((stack_path, label_img[i]))
    return depth_groups

def main(args):
    # Ensure reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Prepare datasets
    train_stack, train_label_img, val_stack, val_label_img = Dataloader.dataloader(args.datapath)
    
    # Group samples by stack depth
    train_depth_groups = group_samples_by_depth(train_stack, train_label_img)
    val_depth_groups = group_samples_by_depth(val_stack, val_label_img)
    
    print("Training data depth groups:", list(train_depth_groups.keys()))
    print("Validation data depth groups:", list(val_depth_groups.keys()))
    
    # Create separate data loaders for each depth group
    train_loaders = {}
    for depth, samples in train_depth_groups.items():
        # Extract stack paths and label paths
        stack_paths = [sample[0] for sample in samples]
        label_paths = [sample[1] for sample in samples]
        
        # Create dataset for this depth group
        depth_dataset = Dataloader.myImageFloder(stack_paths, label_paths, target_size=tuple(args.image_size))
        
        # Create data loader
        train_loaders[depth] = torch.utils.data.DataLoader(
            depth_dataset,
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True
        )

    # Create separate data loaders for each depth group in validation
    val_loaders = {}
    for depth, samples in val_depth_groups.items():
        # Extract stack paths and label paths
        stack_paths = [sample[0] for sample in samples]
        label_paths = [sample[1] for sample in samples]
        
        # Create dataset for this depth group
        depth_dataset = Dataloader.myImageFloder(stack_paths, label_paths, target_size=tuple(args.image_size))
        
        # Create data loader
        val_loaders[depth] = torch.utils.data.DataLoader(
            depth_dataset,
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
        )

    model_save_path = os.path.join('./experiments', args.exp_name)
    os.makedirs(model_save_path, exist_ok=True)
    
    writer = SummaryWriter(log_dir=model_save_path)

    # Define model
    model = StackMFF()
    
    # Set device(s) to use
    if args.gpu_id is None:
        # Use all available GPUs
        if torch.cuda.is_available():
            model = nn.DataParallel(model)
            print(f"Using all available GPUs: {torch.cuda.device_count()}")
        else:
            print("CUDA is not available, using CPU")
    else:
        # Use specified GPU(s)
        if torch.cuda.is_available():
            if len(args.gpu_id) == 1:
                # Single GPU
                device = torch.device(f'cuda:{args.gpu_id[0]}')
                model = model.to(device)
                print(f"Using GPU: {args.gpu_id[0]}")
            else:
                # Multiple specified GPUs
                model = nn.DataParallel(model, device_ids=args.gpu_id)
                print(f"Using GPUs: {args.gpu_id}")
        else:
            print("CUDA is not available, using CPU")

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.0001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    if torch.cuda.is_available():
        model.cuda()

    # Define loss functions once, outside the training loop
    criterion1 = nn.L1Loss()
    criterion2 = loss.LpLssimLoss()
    criterion3 = loss.LaplacianLoss()
    criterion4 = loss.SpatialFrequencyLoss()

    def train(img_stack, img_label):
        if torch.cuda.is_available():
            img_stack, img_label = img_stack.cuda(), img_label.cuda()

        # Use consistent index processing
        img_label = torch.unsqueeze(torch.squeeze(img_label, 1)[:, 0, :, :], 1)  # Remove padding
        
        # Manual normalization since DataLoader normalization is ineffective due to padding
        img_label = img_label / 255.0
        img_label = img_label.float()
        output = model(img_stack)

        loss1 = criterion1(output, img_label)
        loss2 = criterion2(output, img_label)
        loss3 = criterion3(output, img_label)
        loss4 = criterion4(output, img_label)

        # Use parameterized loss weights
        loss = args.loss_weights[0] * loss1 + args.loss_weights[1] * loss2 + args.loss_weights[2] * loss3 + args.loss_weights[3] * loss4

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def test(img_stack, img_label):
        with torch.no_grad():
            if torch.cuda.is_available():
                img_stack, img_label = img_stack.cuda(), img_label.cuda()

            # Use consistent index processing
            img_label = torch.unsqueeze(torch.squeeze(img_label, 1)[:, 0, :, :], 1)  # Remove padding
            
            # Manual normalization since DataLoader normalization is ineffective due to padding
            img_label = img_label / 255.0
            img_label = img_label.float()

            output = model(img_stack)
            loss1 = criterion1(output, img_label)
            loss2 = criterion2(output, img_label)
            loss3 = criterion3(output, img_label)
            loss4 = criterion4(output, img_label)
            # Use parameterized loss weights
            loss = args.loss_weights[0] * loss1 + args.loss_weights[1] * loss2 + args.loss_weights[2] * loss3 + args.loss_weights[3] * loss4
            return loss

    best_val_loss = float('inf')
    for epoch in tqdm(range(0, args.epochs)):
        print('This is %d-th epoch,' % (epoch), 'lr is ', optimizer.param_groups[0]["lr"])
        lr_current = scheduler.get_last_lr()[0]
        writer.add_scalar('lr', lr_current, epoch)
        
        # Training
        model.train()
        total_train_loss = 0
        train_batches = 0
        
        # Train on each depth group
        for depth, train_loader in train_loaders.items():
            tqdm_bar_train = tqdm(train_loader, desc=f'Training (depth={depth})')
            for batch_idx, (img_train_stack, img_train_label) in enumerate(tqdm_bar_train):
                train_loss = train(img_train_stack, img_train_label)
                total_train_loss += train_loss.item()
                train_batches += 1
                
                tqdm_bar_train.set_description(
                    f'Epoch {epoch}, Depth {depth}, Step {batch_idx}, Train Loss {train_loss.item():.4f}, Lr {lr_current}')

        avg_train_loss = total_train_loss / train_batches if train_batches > 0 else 0
        print('Epoch %d total training loss = %.3f' % (epoch, avg_train_loss))
        
        # Log training metrics
        writer.add_scalar('training loss(epoch)', avg_train_loss, epoch)

        # Validation
        model.eval()
        total_val_loss = 0
        val_batches = 0
        
        # Validate on each depth group
        for depth, val_loader in val_loaders.items():
            tqdm_bar_val = tqdm(val_loader, desc=f'Validation (depth={depth})')
            for batch_idx, (img_val_stack, img_val_label) in enumerate(tqdm_bar_val):
                val_loss = test(img_val_stack, img_val_label)
                total_val_loss += val_loss.item()
                val_batches += 1
                tqdm_bar_val.set_description(f'Epoch {epoch}, Depth {depth}, Step {batch_idx}, Val Loss {val_loss.item():.4f}, Lr {lr_current}')

        avg_val_loss = total_val_loss / val_batches if val_batches > 0 else 0
        print('Epoch %d total validation loss = %.3f' % (epoch, avg_val_loss))
        
        # Log validation metrics
        writer.add_scalar('val loss(epoch)', avg_val_loss, epoch)
        
        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Remove 'module.' prefix from state_dict keys if present
            state_dict = model.state_dict()
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}  # Remove 'module.' prefix
            torch.save(state_dict, "{}/best_model.pth".format(model_save_path))
            print('Best model saved in {}'.format(os.path.join(model_save_path, 'best_model.pth')))

        # Save model
        if (epoch + 1) % args.model_save_fre == 0:
            # Remove 'module.' prefix from state_dict keys if present
            state_dict = model.state_dict()
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}  # Remove 'module.' prefix
            torch.save(state_dict, "{}/epoch_{}.pth".format(model_save_path, str(epoch)))
            print('Model saved in {}'.format(os.path.join(model_save_path, 'epoch_%d.pth' % (epoch))))

        scheduler.step()

    writer.close()
    print('Full training time = %.2f HR' % ((time.time() - start_full_time) / 3600))

if __name__ == '__main__':
    main(args)