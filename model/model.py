import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torch


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class Multi_scale_Conv(nn.Module):
    """Temporal-Frequency Multi-scale Convolution MODULE
    Extract temporal and frequency features of single music cilp
    input:(batch_Size,seq_len,channel,Height,Width)
    output:(batch_Size,seq_len,channel,Height//4,Width//8)"""
    def __init__(self):
        super(Multi_scale_Conv, self).__init__()
        self.batch_norm = nn.BatchNorm2d(num_features=1)

        # 分支 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 5), padding=(0, 2))
        self.batch_norm1 = nn.BatchNorm2d(num_features=8)
        self.relu1 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # 分支 2
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 1), padding=(1, 0))
        self.batch_norm2 = nn.BatchNorm2d(num_features=8)
        self.relu2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # 分支 3
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5, 5), padding=(2, 2))
        self.batch_norm3 = nn.BatchNorm2d(num_features=16)
        self.relu3 = nn.ReLU()
        self.max_pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # 分支 4
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=(1, 1))
        self.batch_norm4 = nn.BatchNorm2d(num_features=16)
        self.relu4 = nn.ReLU()
        self.max_pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # 分支 5
        self.conv5 = nn.Conv2d(in_channels=16 * 2, out_channels=64, kernel_size=(3, 3), padding=(1, 1))
        self.batch_norm5 = nn.BatchNorm2d(num_features=64)
        self.relu5 = nn.ReLU()
        self.max_pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # 分支 6
        self.conv6 = nn.Conv2d(in_channels=16 * 2, out_channels=64, kernel_size=(5, 5), padding=(2, 2))
        self.batch_norm6 = nn.BatchNorm2d(num_features=64)
        self.relu6 = nn.ReLU()
        self.max_pool6 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.shape
        x = x.reshape(batch_size * num_frames, channels, height, width)
        # 分支 1
        x1 = self.conv1(x)
        x1 = self.batch_norm1(x1)
        x1 = self.relu1(x1)
        x1 = self.max_pool1(x1)

        # 分支 2
        x2 = self.conv2(x)
        x2 = self.batch_norm2(x2)
        x2 = self.relu2(x2)
        x2 = self.max_pool2(x2)


        x_combined = torch.cat((x1, x2), dim=2)

        # 分支 3
        x3 = self.conv3(x_combined)
        x3 = self.batch_norm3(x3)
        x3 = self.relu3(x3)
        x3 = self.max_pool3(x3)

        # 分支 4
        x4 = self.conv4(x_combined)
        x4 = self.batch_norm4(x4)
        x4 = self.relu4(x4)
        x4 = self.max_pool4(x4)

        x_combined_final = torch.cat((x3, x4), dim=1)
        # 分支 5
        x_A = self.conv5(x_combined_final)
        x_A = self.batch_norm5(x_A)
        x_A = self.relu5(x_A)
        x_A = self.max_pool5(x_A)

        # 分支 6
        x_B = self.conv6(x_combined_final)
        x_B = self.batch_norm6(x_B)
        x_B = self.relu6(x_B)
        x_B = self.max_pool6(x_B)

        x_A = x_A.reshape(batch_size, num_frames, -1, x_A.size(2), x_A.size(3))
        x_B = x_B.reshape(batch_size, num_frames, -1, x_B.size(2), x_B.size(3))
        return x_A, x_B

class ChannelAttention(nn.Module):
    """Channel Attention
    Extract channel features of single music cilp
    input:(batch_Size,seq_len,channel,Height,Width)
    output:(batch_Size,seq_len,channel,1,1)"""
    def __init__(self, in_channels,reduction_ratio=4):
        super(ChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.point_conv1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channels // reduction_ratio)
        self.relu = nn.ReLU()
        self.point_conv2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        _, C, H, W = x.shape
        x=x.view(x.size(0), C, H, W)
        squeezed = self.global_avg_pool(x)
        squeezed = squeezed.view(x.size(0), C, 1, 1)
        excited = self.point_conv1(squeezed)
        excited = self.bn1(excited)
        excited = self.relu(excited)
        excited = self.point_conv2(excited)
        excited = self.bn2(excited)
        excited = excited.view(x.size(0), C, 1, 1)
        return excited


class SpatialAttention(nn.Module):
    """Spatial Attention
    Extract spitial features of single music cilp
    input:(batch_Size,seq_len,channel,Height,Width)
    output:(batch_Size,seq_len,1,Height,Width)"""
    def __init__(self, C, H, W,reduction_ratio=4):
        super(SpatialAttention, self).__init__()
        self.C = C
        self.H = H
        self.W = W
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.point_conv1 = nn.Conv1d(in_channels=H * W, out_channels=(H * W) // reduction_ratio, kernel_size=1)
        self.batch_norm1 = nn.BatchNorm1d((H * W) // reduction_ratio)
        self.relu1 = nn.ReLU()
        self.point_conv2 = nn.Conv1d(in_channels=(H * W) // reduction_ratio, out_channels=H * W, kernel_size=1)
        self.batch_norm2 = nn.BatchNorm1d(H * W)

    def forward(self, x):
        _, C, H, W = x.shape
        x = x.view(x.size(0), C, H * W)
        x = x.view(x.size(0), H * W,C)
        x = self.global_avg_pool(x)
        x = self.point_conv1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.point_conv2(x)
        x = self.batch_norm2(x)
        x = x.view(x.size(0), 1, H*W)
        x = x.view(x.size(0), 1, H, W)
        return x


class DAFF(nn.Module):
    """Dual Attention Feature Fusion
    Extract channel and spitial features of single music cilp
    input:(batch_Size,seq_len,channel,Height,Width)
    output:(batch_Size,seq_len,channel,Height,Width)"""
    def __init__(self,reduction_ratio=4):
        super(DAFF, self).__init__()
        self.channel_attention = ChannelAttention(in_channels=64,reduction_ratio=reduction_ratio)
        self.spatial_attention = SpatialAttention(C=64, H=12, W=16,reduction_ratio=reduction_ratio)
    def forward(self, A, B):
        assert A.shape == B.shape, "A and B must have the same shape"
        batch_size,segments,channel,H,W=A.shape
        A=A.view(batch_size*segments,channel,H,W)
        B =B.view(batch_size * segments, channel, H, W)
        C = torch.add(A, B)
        T1 = self.channel_attention(C)
        T2 = self.spatial_attention(C)
        T1_T2_combined = torch.add(T2, T1)
        D = torch.sigmoid(T1_T2_combined)
        X_prime = A*D
        _X_prime = B*(1-D)
        out=torch.add(X_prime,_X_prime)
        out=out.view(batch_size,segments,out.size(1),out.size(2),out.size(3))
        return out

class Seq_learning(nn.Module):
    """Sequence learning Module
    Use 30s music features(60*0.5s segments) for sequence learning and predict v,a value
    input:(batch_Size,seq_len,channel,Height,Width)
    output:(batch_Size,seq_len,2)"""
    def __init__(self):
        super(Seq_learning, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1))
        self.bilstm1 = nn.LSTM(input_size=12 * 16, hidden_size=48, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(96, 2)

    def forward(self, x):
        batch_size, num_frames, C, H, W = x.shape
        input = x.view(batch_size * num_frames, C, H, W)
        input= self.conv1(input)
        input = input.view(batch_size, num_frames, H * W)
        input, _ = self.bilstm1(input)
        output = self.fc(input)
        output = output.reshape(batch_size, num_frames, 2)
        return output



class DAMFF(BaseModel):
    """Overall DAMFF model
    input:(batch_Size,seq_len,channel,Height,Width)
    output:(batch_Size,seq_len,2)"""
    def __init__(self):
        super(DAMFF, self).__init__()
        self.Mul_sca_Conv_model=Multi_scale_Conv()
        self.Daff_model=DAFF()
        self.seq_learn_model=Seq_learning()

    def forward(self,x):
        output_A, output_B=self.Mul_sca_Conv_model(x)
        X_prime = self.Daff_model(output_A, output_B)
        out=self.seq_learn_model(X_prime)
        out=out.view(out.size(0),out.size(1),-1)
        return out