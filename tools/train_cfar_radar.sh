NUMBA_ENABLE_CUDASIM=1 python3 tools/train.py \
--cfg_file /seeing_beyond/tools/cfgs/kitti_models/cfar-radar.yaml \
--pretrained_model /seeing_beyond/ckpts/cfar/cfar-radar.pth \
--batch_size 32 \
--epochs 40 \
--workers 8 