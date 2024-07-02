python ../main.py\
  --data /home/pc/utils/datasets\
  --lr 0.15 -p 194 --epochs 200 \
  --arch resnet32 \
  --wd 5e-4 \
  --cl_views uncutout-sim \
  --batch-size 256\
  --warmup_epochs 5\
  --feat_dim 128\
  --alpha 2\
  --beta 0.6\
  --temp 0.1\
  --tau 1\
  --root_log "../log"\
  --dataset cifar10\
  --imb_factor 0.1\
  --l_d_warm 100\
  --topk 3\
  --scaling_factor 2 255\

