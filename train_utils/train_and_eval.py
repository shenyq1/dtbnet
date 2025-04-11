import torch
from torch import nn
import math
import train_utils.distributed_utils as utils
from .dice_coefficient_loss import dice_loss, build_target
from torch.nn import functional as F
import train_utils.distributed_utils as utils
from torch.nn.modules.loss import *
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from skimage import morphology
import numpy as np  
from scipy import ndimage 
from torch.nn import CrossEntropyLoss 

def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
 
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

# Dice损失函数
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5
    
    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)   
        pre = torch.sigmoid(predict).view(num, -1)
        tar = target.view(num, -1)      
        intersection = (pre * tar).sum(-1).sum()  #利用预测值与标签相乘当作交集
        union = (pre + tar).sum(-1).sum()       
        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)       
        return score
    
class EnhancedGapLoss(nn.Module):  
    def __init__(self, K=20, direction_weight=0.5, continuity_weight=0.3):  
        super(EnhancedGapLoss, self).__init__()  
        self.K = K  
        self.direction_weight = direction_weight  
        self.continuity_weight = continuity_weight  
        
        # 预定义方向卷积核  
        self.direction_kernels = self._create_direction_kernels()  

    def _create_direction_kernels(self):  
        # 创建方向卷积核，确保大小为奇数以便于padding  
        kernels = []  
        # 水平方向  
        kernels.append(torch.tensor([[1, 1, 1]], dtype=torch.double).view(1, 1, 1, 3))  
        # 垂直方向  
        kernels.append(torch.tensor([[1], [1], [1]], dtype=torch.double).view(1, 1, 3, 1))  
        # 对角线方向  
        kernels.append(torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.double).view(1, 1, 3, 3))  
        kernels.append(torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=torch.double).view(1, 1, 3, 3))  
        return kernels  

    def _compute_continuity_loss(self, skeleton):  
        continuity_loss = 0  
        for kernel in self.direction_kernels:  
            kernel = kernel.to(skeleton.device).double()  
            
            # 计算适当的padding  
            pad_h = (kernel.shape[2] - 1) // 2  
            pad_w = (kernel.shape[3] - 1) // 2  
            
            # 检测连续性  
            response = F.conv2d(  
                skeleton.double(),  
                kernel,  
                padding=(pad_h, pad_w)  
            )  
            
            # 确保尺寸匹配  
            if response.shape != skeleton.shape:  
                response = F.pad(  
                    response,  
                    (0, skeleton.shape[3] - response.shape[3],  
                     0, skeleton.shape[2] - response.shape[2])  
                )  
            
            # 计算损失  
            continuity_loss += torch.mean(torch.abs(response - skeleton))  
        
        return continuity_loss  

    def _compute_direction_consistency(self, skeleton):  
        direction_loss = 0  
        for kernel in self.direction_kernels:  
            kernel = kernel.to(skeleton.device).double()  
            
            # 计算适当的padding  
            pad_h = (kernel.shape[2] - 1) // 2  
            pad_w = (kernel.shape[3] - 1) // 2  
            
            # 计算方向响应  
            response = F.conv2d(  
                skeleton.double(),  
                kernel,  
                padding=(pad_h, pad_w)  
            )  
            
            # 确保尺寸匹配  
            if response.shape != skeleton.shape:  
                response = F.pad(  
                    response,  
                    (0, skeleton.shape[3] - response.shape[3],  
                     0, skeleton.shape[2] - response.shape[2])  
                )  
            
            direction_loss += torch.mean(torch.abs(1 - response))  
        
        return direction_loss  

    def _generate_adaptive_weight_map(self, skeleton, endpoints):  
        skeleton_np = skeleton.cpu().numpy()  
        
        distance_maps = []  
        for i in range(skeleton_np.shape[0]):  
            distance_map = ndimage.distance_transform_edt(1 - skeleton_np[i])  
            distance_maps.append(distance_map)  
        
        distance_map = np.stack(distance_maps, axis=0)  
        distance_map = torch.from_numpy(distance_map).to(skeleton.device)  
        
        weight_map = torch.exp(-distance_map / self.K)  
        endpoints_np = endpoints.cpu().numpy()  
        weight_map = weight_map + torch.from_numpy(endpoints_np).to(skeleton.device) * self.K  
        
        return weight_map.double()  

    def forward(self, pred, target):  
        # 基础交叉熵损失  
        criterion = CrossEntropyLoss(reduction='none')  
        L = criterion(pred, target)  

        # 二值化预测结果  
        A = torch.argmax(pred, dim=1)  

        # 提取骨架  
        A_np = A.cpu().numpy()  
        B = np.zeros_like(A_np)  
        for n in range(A_np.shape[0]):  
            temp = morphology.skeletonize(A_np[n])  
            B[n] = np.where(temp, 1, 0)  
        B = torch.from_numpy(B).to(pred.device).double()  
        B = torch.unsqueeze(B, dim=1)  

        # 检测端点和交叉点  
        kernel = torch.ones((1, 1, 3, 3), dtype=torch.double).to(pred.device)  
        kernel[0][0][1][1] = 0  
        C = F.conv2d(B, weight=kernel, padding=1)  
        C = torch.mul(B, C)  
        
        # 端点和交叉点  
        endpoints = torch.where((C == 1) | (C >= 3), 1.0, 0.0).double()  

        # 生成自适应权重图  
        W = self._generate_adaptive_weight_map(B.squeeze(1), endpoints.squeeze(1))  
        W = W.unsqueeze(1)  

        # 计算损失  
        continuity_loss = self._compute_continuity_loss(B)  
        direction_loss = self._compute_direction_consistency(B)  

        # 组合所有损失  
        base_loss = torch.mean(W * L)  
        total_loss = base_loss + \
                    self.continuity_weight * continuity_loss + \
                    self.direction_weight * direction_loss  

        return total_loss  
    
class GapLoss(nn.Module):
    def __init__(self, K=20):
        super(GapLoss, self).__init__()
        self.K = K

    def forward(self, pred, target):
        # Input is processed by softmax function to acquire cross-entropy map L
        criterion = CrossEntropyLoss(reduction='none')
        L = criterion(pred, target)

        # Input is binarized to acquire image A
        A = torch.argmax(pred, dim=1)

        # Skeleton image B is obtained from A
        A_np = A.cpu().numpy()
        B = np.zeros_like(A_np)
        for n in range(A_np.shape[0]):
            temp = morphology.skeletonize(A_np[n])
            temp = np.where(temp == True, 1, 0)
            B[n] = temp
        B = torch.from_numpy(B).to(pred.device).double()
        B = torch.unsqueeze(B, dim=1)

        # Generate endpoint map C :交点以及端点
        kernel = torch.ones((1, 1, 3, 3), dtype=torch.double).to(pred.device)
        kernel[0][0][1][1] = 0
        C = F.conv2d(B, weight=kernel, bias=None, stride=1, padding=1, dilation=1, groups=1)
        C = torch.mul(B, C)
        D = torch.where(C == 1, 1, 0).double()
        E = torch.where(C >= 3, 1, 0).double()
        C = D + E

        # Generate weight map W ：交点和端点的缓冲区，重点关注的区域
        kernel = torch.ones((1, 1, 5, 5), dtype=torch.double).to(pred.device)
        N = F.conv2d(C, weight=kernel, bias=None, stride=1, padding=2, dilation=1, groups=1)#缓冲区结果
        N = N * self.K
        temp = torch.where(N == 0, 1, 0)
        W = N + temp
        loss = torch.mean(W * L)
        return loss

class roadLoss(torch.nn.Module):
    def __init__(self, theta0=3, theta=3):#3是内外各阔1格，5是内外各阔2格 theta0是内阔，theta是外阔 
        super().__init__()

        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, gt):
        """
        """
        n, c, h, w = pred.shape
        # softmax so that predicted map can be distributed in [0, 1]
        pred = torch.sigmoid(pred)
        #pred四舍五入转换到0和1
        pred = torch.round(pred).numpy().astype(np.uint8)
        # one-hot vector of ground truth
        one_hot_gt = gt.numpy().astype(np.uint8)#真实地表，二分类0和1

        skel_pred = np.zeros_like(pred)
        skel_gt = np.zeros_like(one_hot_gt) 
        for i in range(n):
            skeletonpred =  morphology.skeletonize(pred[i,0,:,:]).astype(np.uint8)
            skeletongt = morphology.skeletonize(one_hot_gt[i,0,:,:]).astype(np.uint8)
            skel_pred[i,0,:,:] = skeletonpred
            skel_gt[i,0,:,:] = skeletongt
        skel_pred = torch.from_numpy(skel_pred).to(pred.device).double()
        skel_gt = torch.from_numpy(skel_gt).to(pred.device).double()
        kernel = torch.ones((1, 1, 3, 3), dtype=torch.double).to(pred.device)
        kernel[0][0][1][1] = 0
        f1 = F.conv2d(skel_pred, weight=kernel, bias=None, stride=1, padding=1, dilation=1, groups=1)
        f1 = torch.mul(skel_pred, f1)
        endpt_pred = torch.where(f1 == 1, 1, 0).double()
        intersecpt_pred = torch.where(f1 >= 3, 1, 0).double()

        f2 = F.conv2d(skel_gt, weight=kernel, bias=None, stride=1, padding=1, dilation=1, groups=1)
        f2 = torch.mul(skel_gt, f2)
        endpt_gt = torch.where(f2 == 1, 1, 0).double()
        intersecpt_gt = torch.where(f2 >= 3, 1, 0).double()
        
        kernel2 = torch.ones((1, 1, 5, 5), dtype=torch.double).to(pred.device)#缓冲区5*5之间检测是否存在交点和端点，哨兵影像5*5已经非常大了，可以按照影像适当调整
        endpt_gt_buf = F.conv2d(endpt_gt, weight=kernel, bias=None, stride=1, padding=2, dilation=1, groups=1)
        intersecpt_gt_buf = F.conv2d(intersecpt_gt, weight=kernel, bias=None, stride=1, padding=2, dilation=1, groups=1)
        Inum = torch.sum(torch.mul(intersecpt_pred, intersecpt_gt_buf))/torch.sum(intersecpt_pred)#用户精度
        Enum = torch.sum(torch.mul(endpt_pred, endpt_gt_buf))/torch.sum(endpt_pred)#用户精度
        loss = 1 - (Inum + Enum)/2
        return loss

def criterion(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True, ignore_index: int = -100,losstype = "CE"):
    losses = {}
    for name, x in inputs.items():
        if name == 'mid':
            continue
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
        # loss = nn.functional.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight)
        if losstype == "gaploss":
            criterion = EnhancedGapLoss()
            loss = criterion(x, target)
        elif losstype == "dice":
            dice_target = build_target(target, num_classes, ignore_index)
            loss = dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
        else :
            loss = nn.functional.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight)
        # criterion = roadLoss()
        # loss3 = criterion(x, target)
        if dice is True and losstype == "gaploss":
            dice_target = build_target(target, num_classes, ignore_index)
            loss += dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
        losses[name] = loss

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + losses['aux']


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)['out']
            confmat.update(target.flatten(), output.argmax(1).flatten())
            dice.update(output, target)

        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()

    return confmat, dice.value.item()


def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes,
                    lr_scheduler, print_freq=10, scaler=None, losstype = "gaploss"):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    if num_classes == 2:
        # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
        loss_weight = torch.as_tensor([1.0, 3.0], device=device)
    else:
        loss_weight = None

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target, loss_weight, num_classes=num_classes, ignore_index=255,losstype = losstype)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
