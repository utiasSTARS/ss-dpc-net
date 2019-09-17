import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np

class joint_model(nn.Module):
    def __init__(self, num_img_channels=8, nb_ref_imgs=1, output_exp=False, dropout_prob=0, mode='online'): 
        super(joint_model, self).__init__()
        self.nb_ref_imgs = nb_ref_imgs
        self.output_exp = output_exp
        self.dropout_prob=dropout_prob
        
        if mode == 'online':
            conv_planes =   [8, 16, 32, 64, 128, 256, 256] #
            upconv_planes = [256, 128, 64, 32, 16, 8, 2*4]
            fc1_multiplier = 6
        if mode == 'offline':
            conv_planes = [2*8, 2*16, 2*32, 2*64, 2*128, 2*256, 2*256]
            upconv_planes = [2*256, 2*128, 2*64, 2*32, 2*16, 2*8, 2*4]
            fc1_multiplier = 15
        self.conv1 = self.downsample_conv(num_img_channels, conv_planes[0], kernel_size=7)
        self.conv2 = self.downsample_conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = self.downsample_conv(conv_planes[1], conv_planes[2])
        self.conv4 = self.downsample_conv(conv_planes[2], conv_planes[3])
        self.conv5 = self.downsample_conv(conv_planes[3], conv_planes[4])
        self.conv6 = self.downsample_conv(conv_planes[4], conv_planes[5])
        self.conv7 = self.downsample_conv(conv_planes[5], conv_planes[6])
        
        self.fc1 = torch.nn.Linear(conv_planes[-1]*fc1_multiplier, int(conv_planes[-1])) #*15 for large image, *6 for reg. img size
        self.fc2 = torch.nn.Linear(conv_planes[-1], int(conv_planes[-1]/2))
        self.fc3 = torch.nn.Linear(int(conv_planes[-1]/2), 6)
        self.T_fc1 = torch.nn.Linear(6, int(conv_planes[-1])) #1536 using 256 for last 3 planes *6 for reg. img size
        self.T_fc2 = torch.nn.Linear(conv_planes[-1], int(conv_planes[-1]/2))
        self.fc = torch.nn.Linear(conv_planes[-1],6)
        self.dropout = torch.nn.Dropout(p=self.dropout_prob)        
 
        self.depth_upconv7 = self.upconv(conv_planes[6],   upconv_planes[0])    # 128,128
        self.depth_upconv6 = self.upconv(upconv_planes[0], upconv_planes[1])    # 128,128
        self.depth_upconv5 = self.upconv(upconv_planes[1], upconv_planes[2])    # 128,64
        self.depth_upconv4 = self.upconv(upconv_planes[2], upconv_planes[3])    # 64, 32
        self.depth_upconv3 = self.upconv(upconv_planes[3], upconv_planes[4])    # 32, 16
        self.depth_upconv2 = self.upconv(upconv_planes[4], upconv_planes[5])    # 16,8
        self.depth_upconv1 = self.upconv(upconv_planes[5], upconv_planes[6])    # 8, 8
        

        self.mask_upconv5 = self.upconv(upconv_planes[1], upconv_planes[2])
        self.mask_upconv4 = self.upconv(upconv_planes[2], upconv_planes[3])
        self.mask_upconv3 = self.upconv(upconv_planes[3], upconv_planes[4])
        self.mask_upconv2 = self.upconv(upconv_planes[4], upconv_planes[5])
        self.mask_upconv1 = self.upconv(upconv_planes[5], upconv_planes[6])

        self.iconv7 = self.conv(upconv_planes[0] + conv_planes[5], upconv_planes[0])    # 256, 128
        self.iconv6 = self.conv(upconv_planes[1] + conv_planes[4], upconv_planes[1])    # 256, 128
        self.iconv5 = self.conv(upconv_planes[2] + conv_planes[3], upconv_planes[2])    # 128, 64
        self.iconv4 = self.conv(upconv_planes[3] + conv_planes[2], upconv_planes[3])    # 64, 32
#        self.iconv3 = self.conv(upconv_planes[4] + conv_planes[1], upconv_planes[4])    # 32, 16
#        self.iconv2 = self.conv(upconv_planes[5] + conv_planes[0], upconv_planes[5])    # 16, 8
#        self.iconv1 = self.conv(upconv_planes[6], upconv_planes[6])                     # 8, 8
        self.iconv3 = self.conv(1 + upconv_planes[4] + conv_planes[1], upconv_planes[4])
        self.iconv2 = self.conv(1 + upconv_planes[5] + conv_planes[0], upconv_planes[5])
        self.iconv1 = self.conv(1 + upconv_planes[6], upconv_planes[6])

        self.predict_disp4 = self.predict_disp(upconv_planes[3])
        self.predict_disp3 = self.predict_disp(upconv_planes[4])
        self.predict_disp2 = self.predict_disp(upconv_planes[5])
        self.predict_disp1 = self.predict_disp(upconv_planes[6])
        self.predict_mask = nn.Conv2d(upconv_planes[6], 1, kernel_size=3, padding=1)
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, T21):
        x = torch.cat(x, 1)
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        ###extract pose
        pose_features = out_conv7.view(-1, out_conv7.size(1)*out_conv7.size(2)*out_conv7.size(3))
        pose_features = self.dropout(pose_features)
        pose = self.fc1(pose_features) #768, 128
        pose = self.dropout(pose)   
        pose = self.fc2(pose)   #128,64
        pose = self.dropout(pose)
#        pose = 0.01*self.fc3(pose)

        T21 = self.T_fc1(T21) #6 --> 128
        T21 = self.dropout(T21)
        T21 = self.T_fc2(T21) #128 --> 64
        T21 = self.dropout(T21)
        
        pose = torch.cat((pose,T21),1)
        pose = 0.01*self.fc(pose)

        ### depth prediction
        out_upconv7 = self.crop_like(self.depth_upconv7(out_conv7), out_conv6)
        concat7 = torch.cat((out_upconv7, out_conv6), 1)
        out_iconv7 = self.iconv7(concat7)

        out_upconv6 = self.crop_like(self.depth_upconv6(out_iconv7), out_conv5)
        concat6 = torch.cat((out_upconv6, out_conv5), 1)
        out_iconv6 = self.iconv6(concat6)

        out_upconv5 = self.crop_like(self.depth_upconv5(out_iconv6), out_conv4)
        concat5 = torch.cat((out_upconv5, out_conv4), 1)
        out_iconv5 = self.iconv5(concat5)


        out_upconv4 = self.crop_like(self.depth_upconv4(out_iconv5), out_conv3)
        concat4 = torch.cat((out_upconv4, out_conv3), 1)
        out_iconv4 = self.iconv4(concat4)
        disp4 = self.predict_disp4(out_iconv4) + 0.001
        disp4_up = self.crop_like(nn.functional.interpolate(disp4, scale_factor=2, mode='bilinear'), out_conv2)

        out_upconv3 = self.crop_like(self.depth_upconv3(out_iconv4), out_conv2)
        concat3 = torch.cat((out_upconv3, out_conv2, disp4_up), 1)
        out_iconv3 = self.iconv3(concat3)
        disp3 = self.predict_disp3(out_iconv3) + 0.001
        disp3_up = self.crop_like(nn.functional.interpolate(disp3, scale_factor=2, mode='bilinear'), out_conv1)

        out_upconv2 = self.crop_like(self.depth_upconv2(out_iconv3), out_conv1)
        concat2 = torch.cat((out_upconv2, out_conv1, disp3_up), 1)
        out_iconv2 = self.iconv2(concat2)
        disp2 = self.predict_disp2(out_iconv2) + 0.001
        disp2_up = self.crop_like(nn.functional.interpolate(disp2, scale_factor=2, mode='bilinear'), x)

        out_upconv1 = self.crop_like(self.depth_upconv1(out_iconv2), x)
        concat1 = torch.cat((out_upconv1, disp2_up), 1)
        out_iconv1 = self.iconv1(concat1)
        disp1 = 0.5*(self.predict_disp1(out_iconv1) + 0.001)
        
        if self.output_exp:
            exp_mask_upconv5 = self.crop_like(self.mask_upconv5(out_conv5), out_conv4)
            exp_mask_upconv4 = self.crop_like(self.mask_upconv4(exp_mask_upconv5), out_conv3)
            exp_mask_upconv3 = self.crop_like(self.mask_upconv3(exp_mask_upconv4), out_conv2)
            exp_mask_upconv2 = self.crop_like(self.mask_upconv2(exp_mask_upconv3), out_conv1)
            exp_mask_upconv1 = self.crop_like(self.mask_upconv1(exp_mask_upconv2), x)
            exp_mask1 = torch.sigmoid(self.predict_mask(exp_mask_upconv1))
        else:
            exp_mask1 = None

        return pose, [exp_mask1], [disp1]

    def downsample_conv(self, in_planes, out_planes, kernel_size=3):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=2),            
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(out_planes), #best model uses this..            
        )    
    
    def predict_disp(self, in_planes, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_planes, self.nb_ref_imgs, kernel_size=kernel_size, padding=(kernel_size-1)//2),
            nn.ReLU()
        )
        
    def conv(self, in_planes, out_planes, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_planes)
        )
        
    def upconv(self, in_planes, out_planes, kernel_size=3):
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2, output_padding=1),             
            nn.ReLU(inplace=True),
        )
    
    def crop_like(self, input, ref):
        assert(input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
        return input[:, :, :ref.size(2), :ref.size(3)]
