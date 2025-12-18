"""
CIFAR-10 训练脚本（适配 deeptage.py 中的 SimpleSpikingNet + STDS）
"""

import os
import time
import argparse
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from deeptage import SimpleSpikingNet, compute_stds_loss

# 工具函数
def accuracy(output, target, topk=(1,)):
    """计算 top-k 精度，output: B x C"""
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / output.size(0)).item())
        return res  # list of percentages

def save_checkpoint(state, is_best, outdir, filename='checkpoint.pth.tar'):
    path = os.path.join(outdir, filename)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(outdir, 'model_best.pth.tar')
        torch.save(state, best_path)

# 评估函数
def validate(model, dataloader, device, T):
    model.eval()
    total = 0
    top1_sum = 0.0
    top5_sum = 0.0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss()
    # 用于统计平均发火率（按 time-step 平均）
    total_spikes = 0.0
    total_neurons = 0
    stage_sigma1 = defaultdict(list)  # 如果想记录每个 stage 的 sigma1

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc='Val', leave=False):
            images = images.to(device)
            targets = targets.to(device)
            B = images.size(0)
            # 将静态图像复制 T 次作为时间序列输入（常见做法）
            x = images.unsqueeze(1).repeat(1, T, 1, 1, 1)  # B x T x C x H x W
            p_t, aux_preds = model(x)

            # 主预测：对 T 个 time-step 的 logits 做平均
            # p_t 是 list 长度 T，项为 B x C
            logits_avg = sum(p_t) / len(p_t)  # B x C
            loss = 0.0
            # 计算 STDS 的 loss（仅用于报告，不用于更新）
            loss = compute_stds_loss(p_t, aux_preds, targets).item()

            # 统计 top1/top5
            top1, top5 = accuracy(logits_avg, targets, topk=(1,5))
            top1_sum += top1 * B / 100.0
            top5_sum += top5 * B / 100.0
            total += B
            loss_sum += loss * B

            # 统计发火率（粗略）：统计每个 time-step 的 spike 比例（如果模型返回了 spikes，我们可以更精确）
            # 我们没有直接得到 spikes 这里只算占位；如果你需要精确发火率，请修改模型 forward 返回 spikes
            # total_spikes += ... ; total_neurons += B * num_neurons * T

    avg_loss = loss_sum / total
    avg_top1 = (top1_sum / total) * 100.0
    avg_top5 = (top5_sum / total) * 100.0
    return avg_loss, avg_top1, avg_top5

# 训练函数
def train_one_epoch(model, train_loader, optimizer, device, epoch, T, scaler=None, log_interval=50):
    model.train()
    ce = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_samples = 0
    correct1 = 0
    total = 0
    start_time = time.time()

    loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}", ncols=120)
    for batch_idx, (images, targets) in loop:
        images = images.to(device)
        targets = targets.to(device)
        B = images.size(0)
        x = images.unsqueeze(1).repeat(1, T, 1, 1, 1)  # B x T x C x H x W

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                p_t, aux_preds = model(x)
                loss = compute_stds_loss(p_t, aux_preds, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            p_t, aux_preds = model(x)
            loss = compute_stds_loss(p_t, aux_preds, targets)
            loss.backward()
            optimizer.step()

        # 统计（以 logits 平均为准）
        logits_avg = sum(p_t) / len(p_t)
        batch_top1 = accuracy(logits_avg, targets, topk=(1,))[0]
        running_loss += float(loss.item()) * B
        running_samples += B
        correct1 += batch_top1 * B / 100.0
        total += B

        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(train_loader):
            loop.set_postfix({
                'Loss': f"{running_loss / running_samples:0.4f}",
                'Top1': f"{(correct1/total)*100:0.2f}",
                'LR': f"{optimizer.param_groups[0]['lr']:.4e}"
            })

    epoch_time = time.time() - start_time
    return running_loss / running_samples, (correct1/total)*100.0, epoch_time

# 主函数
def main():
    parser = argparse.ArgumentParser(description='Train DeepTAGE on CIFAR-10')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--T', default=4, type=int, help='time steps')
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--out-dir', default='checkpoints', type=str)
    parser.add_argument('--resume', default='', type=str, help='checkpoint to resume from')
    parser.add_argument('--use-amp', action='store_true', help='use mixed precision training')
    args = parser.parse_args()

    # 固定随机种子（可选）
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)

    # 数据增强与加载
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # 模型（从 deeptage.py 导入）
    model = SimpleSpikingNet(in_channels=3, num_classes=10, channels=[32, 64, 128], T=args.T, v_th=1.0, decay=2.0, w=0.25)
    model = model.to(device)

    # 优化器与学习率调度（CosineAnnealingLR）
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 混合精度
    scaler = torch.cuda.amp.GradScaler() if (args.use_amp and device == 'cuda') else None

    start_epoch = 0
    best_acc = 0.0
    os.makedirs(args.out_dir, exist_ok=True)

    # 恢复 checkpoint（可选）
    if args.resume:
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint.get('scheduler', scheduler.state_dict()))
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        print(f"=> resumed from epoch {start_epoch}, best_acc {best_acc:.2f}")

    # 主循环
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc, epoch_time = train_one_epoch(model, train_loader, optimizer, device, epoch, args.T, scaler=scaler)
        scheduler.step()

        val_loss, val_top1, val_top5 = validate(model, val_loader, device, args.T)

        is_best = val_top1 > best_acc
        if is_best:
            best_acc = val_top1

        print(f"Epoch {epoch} | Train Loss {train_loss:.4f} Train Acc {train_acc:.2f} | Val Loss {val_loss:.4f} Val Top1 {val_top1:.2f} Val Top5 {val_top5:.2f} | Best {best_acc:.2f} | Time {epoch_time:.1f}s")

        # 写入训练日志
        with open("result/train_log.csv", "a") as f:
            f.write(f"{epoch},{train_loss:.4f},{train_acc:.2f},{val_loss:.4f},{val_top1:.2f},{val_top5:.2f}\n")

        # 保存 checkpoint
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        save_checkpoint(state, is_best, args.out_dir, filename=f'checkpoint_epoch_{epoch}.pth.tar')

    print("训练完成，best accuracy: %.2f" % best_acc)

if __name__ == '__main__':
    main()
