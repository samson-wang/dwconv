import torch
from torch.nn.functional import conv2d
# To simulate backward conv.
k_size = 4
fwd_padding = 3
stride = 1
bwd_padding = k_size - fwd_padding - 1
x = torch.rand((1, 1, 10, 10), requires_grad=True)
w = torch.rand((1, 1, k_size, k_size), requires_grad=True)

x2 = x.detach().clone()[:, :, :-1, :-1]
w2 = w.detach().clone()
x2.requires_grad = True
w2.requires_grad = True

print(x.shape, x.requires_grad)
print(w.shape, w.requires_grad)

y = conv2d(x, w, padding=fwd_padding, stride=stride)
y.retain_grad()
print(y.shape)
y2 = conv2d(x2, w2, padding=fwd_padding, stride=stride)
y2.retain_grad()

loss = 1.0 - y.sum()
loss.backward()

loss2 = 1.0 - y2.sum()
loss2.backward()

print(w.grad, x.grad)
print(y.grad)
print(w)
print(torch.flip(w, [2,3]))

print("Y", y, y2)
print("loss", loss, loss2)
print("Grad 1,2", w.grad, w2.grad)

rot_w = torch.flip(w, [2,3])

# When stride > 1. It's tricky dilation conv. Has to pay attention to size and boundary
x_grad = conv2d(y.grad, rot_w, padding=bwd_padding, stride=1)
w_grad = conv2d(x, y.grad, padding=fwd_padding, stride=1)

#print(x[:, :, :-1:2, :-1:2])
#print('Grad 1,0', (x[:, :, 1:-2:2, :-2:2] * y.grad).sum())

print("Grad", w_grad, w.grad)
#print(x_grad.shape, x.grad.shape, y.shape)
print(w_grad.shape, w.grad.shape)
print("Grad diff", torch.max(abs(w_grad - w.grad)))
print(torch.max(abs(x_grad - x.grad)))
# manual backward
