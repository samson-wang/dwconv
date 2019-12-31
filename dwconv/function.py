import torch
from . import _C
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

# DepthWiseConv2ding

class _DepthWiseConv2d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups):
        #ctx.output_size = _pair(output_size)
        #ctx.spatial_scale = spatial_scale
        #ctx.input_shape = input.size()
        output = _C.depth_wise_conv2d(
            input, weight, bias, stride, padding, dilation, groups
        )
        #ctx.save_for_backward(input, roi, argmax)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        raise
        input, rois, argmax = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        bs, ch, h, w = ctx.input_shape
        grad_input = _C.roi_pool_backward(
            grad_output,
            input,
            rois,
            argmax,
            spatial_scale,
            output_size[0],
            output_size[1],
            bs,
            ch,
            h,
            w,
        )
        return grad_input, None, None, None


depth_wise_conv2d = _DepthWiseConv2d.apply


#class DepthWiseConv2d(nn.Module):
#    def __init__(self, output_size, spatial_scale):
#        super(DepthWiseConv2d, self).__init__()
#        self.output_size = output_size
#        self.spatial_scale = spatial_scale
#
#    def forward(self, input, rois):
#        return roi_pool(input, rois, self.output_size, self.spatial_scale)
#
#    def __repr__(self):
#        tmpstr = self.__class__.__name__ + "("
#        tmpstr += "output_size=" + str(self.output_size)
#        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
#        tmpstr += ")"
#        return tmpstr
