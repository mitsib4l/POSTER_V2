import shutil
import warnings
from scikit-learn import metrics
from scikit-learn.metrics import confusion_matrix
warnings.filterwarnings("ignore")
import torch.utils.data as data
import os
import argparse
from scikit-learn.metrics import f1_score, confusion_matrix
from data_preprocessing.sam import SAM
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import datetime
from torchsampler import ImbalancedDatasetSampler
from models.PosterV2_7cls import *
import random

import os
from torchvision.transforms.functional import to_pil_image
from models.gradcam import generate_and_save_gradcam

from lime import lime_image
import shap
import torch.nn.functional as F
from skimage.segmentation import slic

warnings.filterwarnings("ignore", category=UserWarning)

now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H-%M]-")

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default=r'/home/Dataset/RAF')
parser.add_argument('--data_type', default='RAF-DB',
                    choices=['RAF-DB', 'AffectNet-7', 'CAER-S', 'fer2013'],
                    type=str, help='dataset option')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/' + time_str + 'model.pth')
parser.add_argument('--best_checkpoint_path', type=str, default='./checkpoint/' + time_str + 'model_best.pth')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=144, type=int, metavar='N')
parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')

parser.add_argument('--lr', '--learning-rate', default=0.000035, type=float, metavar='LR', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=30, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to checkpoint')
parser.add_argument('-e', '--evaluate', default=None, type=str, help='evaluate model on test set')
parser.add_argument('--beta', type=float, default=0.6)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    best_acc = 0
    print('Training time: ' + now.strftime("%m-%d %H:%M"))
    
    # Start timing
    import time
    start_time = time.time()
    
    # Check and print device information
    if torch.cuda.is_available():
        print(f'Using device: GPU (CUDA)')
        print(f'GPU Name: {torch.cuda.get_device_name(0)}')
        print(f'Number of GPUs: {torch.cuda.device_count()}')
    else:
        print('Using device: CPU (No CUDA available)')
        print('WARNING: Training on CPU will be extremely slow!')
    
    # Log the command used to run this script
    import sys
    command_used = ' '.join(sys.argv)
    print(f'Command: {command_used}\n')
    
    # Write command to log file
    txt_name = './log/' + time_str + 'log.txt'
    with open(txt_name, 'a') as f:
        f.write('='*80 + '\n')
        f.write(f'Training Started: {now.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Command: {command_used}\n')
        f.write('='*80 + '\n\n')

    # create model with CBAM option
    model = pyramid_trans_expr2(img_size=224, num_classes=7)

    model = torch.nn.DataParallel(model).cuda()

    criterion = torch.nn.CrossEntropyLoss()

    if args.optimizer == 'adamw':
        base_optimizer = torch.optim.AdamW
    elif args.optimizer == 'adam':
        base_optimizer = torch.optim.Adam
    elif args.optimizer == 'sgd':
        base_optimizer = torch.optim.SGD
    else:
        raise ValueError("Optimizer not supported.")

    optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, rho=0.05, adaptive=False, )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    recorder = RecorderMeter(args.epochs)
    recorder1 = RecorderMeter1(args.epochs)

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, weights_only=False)
            args.start_epoch = checkpoint.get('epoch', 0)
            best_acc = checkpoint.get('best_acc', 0.0)
            if 'recorder' in checkpoint:
                recorder = checkpoint['recorder']
            if 'recorder1' in checkpoint:
                recorder1 = checkpoint['recorder1']
            # Expand recorder arrays if needed
            if args.epochs > recorder.total_epoch:
                old_losses = recorder.epoch_losses
                old_acc = recorder.epoch_accuracy
                recorder.epoch_losses = np.zeros((args.epochs, 2), dtype=np.float32)
                recorder.epoch_losses[:old_losses.shape[0]] = old_losses
                recorder.epoch_accuracy = np.zeros((args.epochs, 2), dtype=np.float32)
                recorder.epoch_accuracy[:old_acc.shape[0]] = old_acc
                recorder.total_epoch = args.epochs

                old_losses1 = recorder1.epoch_losses
                old_acc1 = recorder1.epoch_accuracy
                recorder1.epoch_losses = np.zeros((args.epochs, 2), dtype=np.float32)
                recorder1.epoch_losses[:old_losses1.shape[0]] = old_losses1
                recorder1.epoch_accuracy = np.zeros((args.epochs, 2), dtype=np.float32)
                recorder1.epoch_accuracy[:old_acc1.shape[0]] = old_acc1
                recorder1.total_epoch = args.epochs
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> Resuming training from epoch {args.start_epoch}")
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint.get('epoch', 0)})")
    else:
        print(f"=> no checkpoint found at '{args.resume}'")
    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'valid')

    # choose transforms
    if args.data_type in ('fer2013'):
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(scale=(0.02, 0.1))
        ])
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
    else:
        if args.data_type == 'RAF-DB':
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(scale=(0.02, 0.1))
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=1, scale=(0.05, 0.05))
            ])
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

    # create datasets
    train_dataset = None
    if os.path.isdir(traindir):
        train_dataset = datasets.ImageFolder(traindir, train_transform)
    test_dataset = datasets.ImageFolder(valdir, test_transform)

    # create loaders (use ImbalancedDatasetSampler for AffectNet-7)
    if train_dataset is not None:
        if args.data_type == 'AffectNet-7':
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    sampler=ImbalancedDatasetSampler(train_dataset),
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
                                                    num_workers=args.workers,
                                                    pin_memory=True)
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    num_workers=args.workers,
                                                    pin_memory=True)

    val_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=args.workers,
                                            pin_memory=True)

    if args.evaluate is not None:
        if os.path.isfile(args.evaluate):
            print("=> loading checkpoint '{}'".format(args.evaluate))
            checkpoint = torch.load(args.evaluate, weights_only=False)
            best_acc = checkpoint['best_acc']
            print(f'best_acc:{best_acc}')
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.evaluate))
        
        # Log evaluation command
        import sys
        command_used = ' '.join(sys.argv)
        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write('='*80 + '\n')
            f.write(f'Evaluation Started: {now.strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'Command: {command_used}\n')
            f.write(f'Checkpoint: {args.evaluate}\n')
            f.write('='*80 + '\n\n')
        
        validate(val_loader, model, criterion, args)
        return

    matrix = None

    for epoch in range(args.start_epoch, args.epochs):
        # Start epoch timer
        import time as time_module
        epoch_start_time = time_module.time()

        current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        print('Current learning rate: ', current_learning_rate)
        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write('Current learning rate: ' + str(current_learning_rate) + '\n')

        # train for one epoch
        train_acc, train_los = train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        val_acc, val_los, output, target, D = validate(val_loader, model, criterion, args)

        scheduler.step()

        recorder.update(epoch, train_los, train_acc, val_los, val_acc)
        recorder1.update(output, target)

        curve_name = time_str + 'cnn.png'
        recorder.plot_curve(os.path.join('./log/', curve_name))

        # remember best acc and save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        print('Current best accuracy: ', best_acc.item())

        if is_best:
            matrix = D

        print('Current best matrix: ', matrix)
        
        # Calculate and log epoch time
        epoch_time = time_module.time() - epoch_start_time
        epoch_minutes = int(epoch_time // 60)
        epoch_seconds = int(epoch_time % 60)
        print(f'Epoch [{epoch + 1}/{args.epochs}] completed in {epoch_minutes:02d}:{epoch_seconds:02d} ({epoch_time:.2f}s)')

        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write('Current best accuracy: ' + str(best_acc.item()) + '\n')
            f.write(f'Epoch [{epoch + 1}/{args.epochs}] completed in {epoch_minutes:02d}:{epoch_seconds:02d} ({epoch_time:.2f}s)\n')

        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'recorder1': recorder1,
                         'recorder': recorder}, is_best, args)
    
    # Calculate and log total training time
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    time_str_formatted = f'{hours:02d}:{minutes:02d}:{seconds:02d}'
    print(f'\n{"="*50}')
    print(f'Training completed!')
    print(f'Total training time: {time_str_formatted} ({total_time:.2f} seconds)')
    print(f'Final best accuracy: {best_acc.item():.4f}')
    print(f'{"="*50}\n')
    
    txt_name = './log/' + time_str + 'log.txt'
    with open(txt_name, 'a') as f:
        f.write(f'\n{"="*50}\n')
        f.write(f'Training completed!\n')
        f.write(f'Total training time: {time_str_formatted} ({total_time:.2f} seconds)\n')
        f.write(f'Final best accuracy: {best_acc.item():.4f}\n')
        f.write(f'{"="*50}\n')


def train(train_loader, model, criterion, optimizer, epoch, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [losses, top1],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    for i, (images, target) in enumerate(train_loader):
        # print(images.shape)
        images = images.cuda()
        target = target.cuda()

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, _ = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        # optimizer.step()
        optimizer.first_step(zero_grad=True)
        images = images.cuda()
        target = target.cuda()

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, _ = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.second_step(zero_grad=True)

        # print loss and accuracy
        if i % args.print_freq == 0:
            progress.display(i)

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [losses, top1],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    num_classes = getattr(args, 'num_classes', 7)
    D = np.zeros((num_classes, num_classes), dtype=int)

    first_batch_images = None
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()
            output = model(images)
            if i == 0:
                first_batch_images = images.clone().detach()
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc, _ = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc[0], images.size(0))

            topk = (1,)
            # """Computes the accuracy over the k top predictions for the specified values of k"""
            with torch.no_grad():
                maxk = max(topk)
                # batch_size = target.size(0)
                _, pred = output.topk(maxk, 1, True, True)
                pred = pred.t()

            output = pred
            target = target.squeeze().cpu().numpy()
            output = output.squeeze().cpu().numpy()

            im_re_label = np.array(target)
            im_pre_label = np.array(output)
            y_ture = im_re_label.flatten()
            im_re_label.transpose()
            y_pred = im_pre_label.flatten()
            im_pre_label.transpose()

            C = metrics.confusion_matrix(y_ture, y_pred, labels=list(range(num_classes)))
            D += C

            if i % args.print_freq == 0:
                progress.display(i)


    # Grad-CAM, LIME, and SHAP generation 
    if first_batch_images is not None:
        os.makedirs('./log/gradcam', exist_ok=True)
        os.makedirs('./log/lime', exist_ok=True)
        os.makedirs('./log/shap', exist_ok=True)


        num_classes = getattr(args, 'num_classes', 7)
        class_to_indices = {i: [] for i in range(num_classes)}
        all_images = []
        all_targets = []

        # Gather all images and their labels
        for images, targets in val_loader:
            for img, label in zip(images, targets):
                class_to_indices[int(label)].append((img, label))
            # Stop early if all classes have at least one image
            if all(len(v) > 0 for v in class_to_indices.values()):
                break

        # Randomly select one image per class
        selected_samples = []
        for cls in range(num_classes):
            img, label = random.choice(class_to_indices[cls])
            selected_samples.append((img, label))

        # --- Run explainability methods on these images ---
        for idx, (img, label) in enumerate(selected_samples):
            input_tensor = img.unsqueeze(0).cuda() if torch.cuda.is_available() else img.unsqueeze(0)
            pil_img = to_pil_image(img.cpu() * 0.229 + 0.485)
            rgb_img = np.array(pil_img.resize((224,224))).astype(np.float32) / 255.0
            save_prefix = f'./log/gradcam/class_{int(label)}'

            # Grad-CAM
            generate_and_save_gradcam(model, input_tensor, rgb_img, save_prefix, device='cuda' if input_tensor.is_cuda else 'cpu')
            with torch.no_grad():
                out = model(input_tensor)
                logits = out if not isinstance(out, (tuple, list)) else out[0]
                pred_class = int(logits.argmax(dim=1).item())
            print(f"Class {int(label)}: Predicted={pred_class}, Actual={int(label)}, GradCAM: {save_prefix}_ir_back.png, {save_prefix}_conv3.png")

            # LIME
            if lime_image is not None:
                explainer = lime_image.LimeImageExplainer()
                def batch_predict(images_np):
                    images_t = torch.tensor(images_np.transpose((0,3,1,2)), dtype=torch.float32)
                    images_t = images_t.cuda() if input_tensor.is_cuda else images_t
                    images_t = (images_t - 0.485) / 0.229
                    with torch.no_grad():
                        logits = model(images_t)
                        logits = logits if not isinstance(logits, (tuple, list)) else logits[0]
                        probs = F.softmax(logits, dim=1).cpu().numpy()
                    return probs
                segmentation_fn = lambda x: slic(x, n_segments=50, compactness=1, sigma=1)
                explanation = explainer.explain_instance(
                    (rgb_img*255).astype(np.uint8),
                    batch_predict,
                    top_labels=1,
                    hide_color=0,
                    num_samples=1000,
                    segmentation_fn=segmentation_fn
                )
                lime_img, mask = explanation.get_image_and_mask(
                    label=explanation.top_labels[0], positive_only=True, hide_rest=False, num_features=5
                )
                lime_path = f'./log/lime/class_{int(label)}.png'
                from PIL import Image
                Image.fromarray(lime_img).save(lime_path)
                print(f"Class {int(label)}: Predicted={pred_class}, Actual={int(label)}, LIME: {lime_path}")

            # SHAP
            if shap is not None:
                def shap_batch_predict(x):
                    x = torch.tensor(x.transpose((0,3,1,2)), dtype=torch.float32)
                    x = x.cuda() if input_tensor.is_cuda else x
                    x = (x - 0.485) / 0.229
                    with torch.no_grad():
                        logits = model(x)
                        logits = logits if not isinstance(logits, (tuple, list)) else logits[0]
                        return F.softmax(logits, dim=1).cpu().numpy()
                background = np.expand_dims(rgb_img, 0)
                masker = shap.maskers.Image("inpaint_telea", rgb_img.shape)
                explainer = shap.Explainer(shap_batch_predict, masker)
                shap_values = explainer(np.expand_dims(rgb_img, 0))
                shap_path = f'./log/shap/class_{int(label)}.png'
                shap.image_plot(shap_values, np.expand_dims(rgb_img, 0), show=False)
                import matplotlib.pyplot as plt
                plt.savefig(shap_path)
                plt.close()
                print(f"Class {int(label)}: Predicted={pred_class}, Actual={int(label)}, SHAP: {shap_path}")

    print(' **** Accuracy {top1.avg:.3f} *** '.format(top1=top1))
    with open('./log/' + time_str + 'log.txt', 'a') as f:
        f.write(' * Accuracy {top1.avg:.3f}'.format(top1=top1) + '\n')
    print(D)
    return top1.avg, losses.avg, output, target, D

def save_checkpoint(state, is_best, args):
    torch.save(state, args.checkpoint_path)
    if is_best:
        torch.save(state, args.best_checkpoint_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


labels = ['A', 'B', 'C', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']


class RecorderMeter1(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, output, target):
        self.y_pred = output
        self.y_true = target

    def plot_confusion_matrix(self, cm, title='Confusion Matrix', cmap=plt.cm.binary):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        y_true = self.y_true
        y_pred = self.y_pred

        plt.title(title)
        plt.colorbar()
        xlocations = np.array(range(len(labels)))
        plt.xticks(xlocations, labels, rotation=90)
        plt.yticks(xlocations, labels)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        cm = confusion_matrix(y_true, y_pred)
        np.set_printoptions(precision=2)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(12, 8), dpi=120)

        ind_array = np.arange(len(labels))
        x, y = np.meshgrid(ind_array, ind_array)
        for x_val, y_val in zip(x.flatten(), y.flatten()):
            c = cm_normalized[y_val][x_val]
            if c > 0.01:
                plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
        # offset the tick
        tick_marks = np.arange(len(7))
        plt.gca().set_xticks(tick_marks, minor=True)
        plt.gca().set_yticks(tick_marks, minor=True)
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')
        plt.grid(True, which='minor', linestyle='-')
        plt.gcf().subplots_adjust(bottom=0.15)

        plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
        # show confusion matrix
        plt.savefig('./log/confusion_matrix.png', format='png')
        # fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print('Saved figure')
        plt.show()

    def matrix(self):
        target = self.y_true
        output = self.y_pred
        im_re_label = np.array(target)
        im_pre_label = np.array(output)
        y_ture = im_re_label.flatten()
        # im_re_label.transpose()
        y_pred = im_pre_label.flatten()
        im_pre_label.transpose()

class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 30
        self.epoch_losses[idx, 1] = val_loss * 30
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):
        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1800, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 5
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print('Saved figure')
        plt.close(fig)


if __name__ == '__main__':
    main()
