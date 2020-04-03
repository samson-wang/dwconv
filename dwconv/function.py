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
        ctx.input_shape = input.size()
        groups = ctx.input_shape[1]
        ctx.params = (stride, padding, dilation, groups)
        assert groups == ctx.input_shape[1], "Groups {} must equal to input channels {}" . format(groups, ctx.input_shape[1])
        assert groups == weight.size(0), "Groups {} must equal to weight ouput channels {}" . format(groups, weight.size(0))
        assert dilation == 1, "Not support dilation conv"
        output = _C.depth_wise_conv2d_forward(
            input, weight, bias, stride, padding, dilation, groups
        )
        ctx.save_for_backward(input, weight, bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        bs, ch, h, w = ctx.input_shape
        stride, padding, dilation, groups = ctx.params
        grad_input, grad_weight, grad_bias = _C.depth_wise_conv2d_backward(
            grad_output,
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups
        )
        return grad_input, grad_weight, grad_bias, None, None, None, None


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
