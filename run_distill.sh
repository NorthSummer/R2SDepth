#nohup horovodrun -np 3 -H localhost:3 python scripts/train.py configs/train_kitti_spike.yaml
nohup nohup horovodrun -np 4 -H localhost:4 python scripts/train_distill.py configs/train_kitti_spike_distill.yaml 2>&1 | tee log &