import torch

import dwconv._C

random_seed = 9
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

n = 4
c = 480
size = 700
ksize = 3
x = torch.rand((n,c,size,size)).cuda()
w = torch.rand((c,1,ksize,ksize)).cuda()
bias = torch.rand((c)).cuda()
#print(dwconv._C.depth_wise_conv2d(x, w, w, 1, 1, 1, 1))
print(torch.nn.functional.conv2d(x, w, bias, 1, 0, groups=c).shape)
print(dwconv._C.depth_wise_conv2d(x, w, bias, 1, 0, 1, 1).shape)
import time


st = time.time()
dry_num = 5
num_batches = 10

for i in range(dry_num + num_batches):
    if i == dry_num:
        st = time.time()
    o2 = dwconv._C.depth_wise_conv2d(x, w, bias, 1, 0, 1, 1)
o2 = o2.cpu().detach()
et = time.time()

o2_t = et - st
print("My dw", et - st, num_batches / (et - st))


for i in range(dry_num + num_batches):
    if i == dry_num:
        st = time.time()
    o1 = torch.nn.functional.conv2d(x, w, bias, 1, 0, groups=c)
o1 = o1.cpu().detach()
et = time.time()
o1_t = et - st
print("Official", et - st, num_batches / (et - st))

#diff = dwconv._C.depth_wise_conv2d(x, w, torch.empty(0), 1, 1, 1, 1) - torch.nn.functional.conv2d(x, w, None, 1, 1, groups=c)
diff = o2 - o1
print(torch.max(torch.abs(diff)))
print("Speed up", o1_t / o2_t)
