import torch
import torch.nn.functional as F
import torch.nn as nn
import utils.geometry_helpers 

class Reconstructor(nn.Module):
    def __init__(self):
        super(Reconstructor, self).__init__()

    # Spatial transformer network forward function
    def inverse_warp(self, x, depth, pose, intrinsics, intrinsics_inv):
        """Generates a grid of normalized ([-1,1]) pixel coordinates (size [BxHxWx2]) that torch.functional.grid_sample uses as input.
                Args:
                    x: The input image [BxCxHxW]
                    depth: The depth images [BxHxW]
                    pose: [Bx3x4] concatenated rotation/translation between source and target image (predicted)
        """
        B, C, H, W = x.size()
        cam_coords = self.pixel2cam(depth, intrinsics_inv)
        proj_cam_to_src_pixel = intrinsics.bmm(pose)
        src_pixel_coords = self.cam2pixel(cam_coords, proj_cam_to_src_pixel)
        #grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, src_pixel_coords)

        return x

    def pixel2cam(self, depth, intrinsics_inv):
        """Converts the image grid into an array of camera coordinates (x,y,z) values for all pixels in the image) using the camera intrinsics and depth.
                Args:
                        depth: [BxHxW]
                        uses camera inverse intrinsics (expanded to Bx3x3)
                Returns:
                        array of camera coordinates (x,y,z) for every pixel in the image [Bx3xHxW]
        """
        b, _, h, w = depth.size()
        i_range = torch.arange(0, h).view(1, h, 1).expand(1,h,w).type_as(depth)  # [1, H, W]
        j_range = torch.arange(0, w).view(1, 1, w).expand(1,h,w).type_as(depth)  # [1, H, W]
        ones = torch.ones(1,h,w).type_as(depth)
        pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]
            ###pixel_coords is an array of camera pixel coordinates (x,y,1) where x,y origin is the upper left corner of the image.
        current_pixel_coords = pixel_coords[:,:,:h,:w].expand(b,3,h,w).view(b,3,-1) #.contiguous().view(b, 3, -1)  # [B, 3, H*W]
        #cam_coords = intrinsic_inv.expand(b,3,3).bmm(current_pixel_coords).view(b,3,h,w)
        cam_coords = intrinsics_inv.bmm(current_pixel_coords).view(b,3,h,w)
        return cam_coords * depth

    def cam2pixel(self, cam_coords, pose):
        """cam_coords from pixel2cam are transformed (with rot and trans) and convertd back to pixels (and normalized to be within [-1,1])
                Args:
                    cam_coords: [Bx3xHxW]
                    pose: [Bx3x4]
                Returns:
                    array of [-1,1] coordinates [B,H,W,2]
        """

        b, _, h, w = cam_coords.size()
        cam_coords_flat = cam_coords.view(b,3,-1) # [B,3,H*W]
        pcoords = pose[:,:,0:3].bmm(cam_coords_flat) + pose[:,:,3].view(b,3,1)  #Bx[3x3 x 3xH*W] = [B x 3 x H*W]
        X, Y, Z = pcoords[:,0,:].clamp(-1e20,1e20), pcoords[:,1,:].clamp(-1e20,1e20), pcoords[:,2,:].clamp(1e-20,1e20) #each are [B x H*W]        
        X_norm = 2*(X / Z)/(w-1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
        Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]

        X_mask = ((X_norm > 1)+(X_norm < -1)).detach()
        X_norm[X_mask] = 2  # make sure that no point in warped image is a combinaison of im and gray
        Y_mask = ((Y_norm > 1)+(Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

        pixel_coords = torch.stack([X_norm, Y_norm], dim=2) # [B, H*W, 2]
        return pixel_coords.view(b,h,w,2)
    def resize_img(self, x, depth,intrinsics, intrinsics_inv):
            b,_,h,w = depth.size()
            downscale = x.size(2)/h
            img_scaled = nn.functional.adaptive_avg_pool2d(x, (h, w))
            intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)
            intrinsics_scaled_inv = torch.cat((intrinsics_inv[:, :, 0:2]*downscale, intrinsics_inv[:, :, 2:]), dim=2)
            return img_scaled, intrinsics_scaled, intrinsics_scaled_inv

    def forward(self, x, depth, pose, intrinsics, intrinsics_inv):
        # transform the input
        B,C,H,W = x.size()
        trans = pose[:,0:3].unsqueeze(2)
        rot = utils.geometry_helpers.euler2mat(pose[:,3:6])

        pose = torch.cat([rot,trans],dim=2)
#        bottom_row = torch.FloatTensor([0,0,0,1]).view((1,1,4)).repeat(pose.size(0),1,1)
#        pose= torch.cat((pose, bottom_row),dim=1) #pose is now Bx4x4
        
        if x.size(2) != depth.size(2) or x.size(3) != depth.size(3):
            x, intrinsics, intrinsics_inv = self.resize_img(x,depth, intrinsics, intrinsics_inv)
        x = self.inverse_warp(x, depth, pose, intrinsics, intrinsics_inv)
        return x
