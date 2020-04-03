import torch

from dwconv import depth_wise_conv2d

random_seed = 9
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

n = 1
c = 3
size_x = 7
size_y = 9
ksize = 3
padding = 0
stride = 1
def test_time(n, c, size_x, size_y, ksize, stride, padding):
    x = torch.rand((n,c,size_y,size_x)).cuda()
    w = torch.rand((c,1,ksize,ksize)).cuda()
    bias = torch.rand((c)).cuda()
    bias = None
    #print(x)
    #print(dwconv._C.depth_wise_conv2d(x, w, w, 1, 1, 1, 1))
    #print(torch.nn.functional.conv2d(x, w, bias, stride, padding, groups=c).shape)
    #print(depth_wise_conv2d(x, w, bias if bias is not None else torch.empty(0), stride, padding, 1, 1))
    import time
    
    st = time.time()
    dry_num = 5
    num_batches = 10
    with torch.autograd.profiler.profile(enabled=False, use_cuda=True) as prof:
       
        
        for i in range(dry_num + num_batches):
            if i == dry_num:
                st = time.time()
            o1 = torch.nn.functional.conv2d(x, w, bias, stride, padding, groups=c)
        torch.cuda.synchronize()
        #o1 = o1.cpu().detach()
        et = time.time()
        o1_t = et - st
        print("Official", et - st, num_batches / (et - st))
    #print(prof.key_averages().table(sort_by="cuda_time"))
    #diff = dwconv._C.depth_wise_conv2d(x, w, torch.empty(0), 1, 1, 1, 1) - torch.nn.functional.conv2d(x, w, None, 1, 1, groups=c)
    #print(o2[torch.abs(diff) == torch.max(torch.abs(diff))], o1[torch.abs(diff) == torch.max(torch.abs(diff))], torch.nonzero(torch.abs(diff) == torch.max(torch.abs(diff))))
    #print(o2[4][1100:1124], o1[4][1120:1124])
    print("NCHW {} x {} x {} x {}, kernel {}, stride {}, padding {}, Out Shape {}" . format(n, c, size_y, size_x, ksize, stride, padding, o1.shape))
#print(o2 - x)

eff_arg_list = [
                    (16, 32, 112, 112, 112, 1, 1),
                    (16, 144, 56, 56, 56, 1, 1),
                    (16, 240, 28, 28, 28, 1, 2),
                    (16, 480, 14, 14, 14, 1, 1),
                    (16, 480, 14, 14, 14, 1, 2),
                    (16, 672, 14, 14, 14, 1, 2),
                    (16, 1152, 7, 7, 7, 1, 2),
                    (16, 1152, 7, 7, 7, 1, 1)
               ]
#eff_arg_list = [(16, 1152, 7, 7, 3, 1, 1)]
for args in eff_arg_list:
    test_time(*args)
