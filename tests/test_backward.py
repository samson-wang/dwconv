import torch
from torch.nn.functional import conv2d
from dwconv import depth_wise_conv2d

random_seed = 9
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = True

# To simulate backward conv.
n = 1
c = 1
size_x = 34
size_y = 34
k_size = 32
padding = 0
stride = 1

x = torch.rand((n, c, size_y, size_x), requires_grad=True).cuda()
x = torch.arange((n * c * size_y * size_x)).reshape((n, c, size_y, size_x)).float().cuda()
x.requires_grad = True

w = torch.rand((c, 1, k_size, k_size), requires_grad=True).cuda()
#x = torch.ones((n, c, size_y, size_x), requires_grad=True).cuda()
#w = torch.ones((c, 1, k_size, k_size), requires_grad=True).cuda()
x.retain_grad()

x1 = x.detach().clone()
w1 = w.detach().clone()
x1.requires_grad = True
w1.requires_grad = True

y = conv2d(x, w, padding=padding, stride=stride, groups=c)
y.retain_grad()

loss = 1.0 - y.sum()
loss.backward()

y1 = depth_wise_conv2d(x1, w1, torch.empty(0, requires_grad=True).cuda(), stride, padding, 1, 1)
#print(y1, type(y1), y, type(y))
y1.retain_grad()
loss1 = 1.0 - y1.sum()

print("Y", y, y1, y.shape, y1.shape, y - y1)
loss1.backward()

print("LOSS", loss - loss1, loss, loss1)
print("x grad", x.grad, x1.grad)
print("x grad diff", torch.max(abs(x.grad - x1.grad)))
print(x.grad - x1.grad)
# manual backward
