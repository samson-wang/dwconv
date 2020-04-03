# Depth wise convolution in Pytorch

## Forward

Done for stride=1 cases.

```
My dw 0.007719278335571289 1295.4578867714736
Official 0.019470930099487305 513.5861486279648
NCHW 16 x 32 x 112 x 112, kernel 3, stride 1, padding 1
tensor(0., device='cuda:0')
Speed up 2.5223769960156903
My dw 0.008510351181030273 1175.0396414063594
Official 0.013824939727783203 723.3304590763288
NCHW 16 x 144 x 56 x 56, kernel 3, stride 1, padding 1
tensor(0., device='cuda:0')
Speed up 1.6244852220198907
My dw 0.0043735504150390625 2286.4718709114695
Official 0.008342504501342773 1198.6808036352204
NCHW 16 x 240 x 28 x 28, kernel 5, stride 1, padding 2
tensor(0., device='cuda:0')
Speed up 1.907490187527257
My dw 0.0009541511535644531 10480.519740129936
Official 0.0015006065368652344 6663.972036860502
NCHW 16 x 480 x 14 x 14, kernel 3, stride 1, padding 1
tensor(0., device='cuda:0')
Speed up 1.5727136431784108
My dw 0.002266407012939453 4412.270145171471
Official 0.004055976867675781 2465.4972960263344
NCHW 16 x 480 x 14 x 14, kernel 5, stride 1, padding 2
tensor(0., device='cuda:0')
Speed up 1.7896065642751946
My dw 0.0031287670135498047 3196.1472224338945
Official 0.0056688785552978516 1764.0173276695966
NCHW 16 x 672 x 14 x 14, kernel 5, stride 1, padding 2
tensor(0., device='cuda:0')
Speed up 1.8118570448830298
My dw 0.0010640621185302734 9397.947568899843
Official 0.0024340152740478516 4108.437653051229
NCHW 16 x 1152 x 7 x 7, kernel 5, stride 1, padding 2
tensor(0., device='cuda:0')
Speed up 2.2874747927403094
My dw 0.0005471706390380859 18275.834422657954
Official 0.0008418560028076172 11878.516001132824
NCHW 16 x 1152 x 7 x 7, kernel 3, stride 1, padding 1
tensor(0., device='cuda:0')
Speed up 1.538562091503268
```

## Backward

backward is a fractional strided transposed convolution. For input x, it's relative simple. For weight w, it's a large kernel convolution. I've implemented a reference version for stried=1 and padding=0 case. The data reuse pattern is not critical in such case. Less unnecessary computation, More registers!

Speedup looks good when the kernel size is large. When kernels less than 64, the kernel launch configuration should be adjusted.

```
('My dw', 0.023216962814331055, 430.7195596586533)
('Official', 0.058522939682006836, 170.8731662205709)
NCHW 16 x 32 x 112 x 112, kernel 110, stride 1, padding 0
tensor(0.0205, device='cuda:0')
('Speed up', 2.5206974809763913)
```
