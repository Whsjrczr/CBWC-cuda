#!/usr/bin/env bash
cd "$(dirname $0)/.."
python3 mnist_RandomLabel.py \
-a=LinearModel \
--width=512 \
--depth=4 \
--batch-size=256 \
--epochs=300 \
-oo=sgd \
-oc=momentum=0 \
-wd=0 \
--lr=0.9 \
--lr-method=step \
--lr-step=1 \
--lr-gamma=1 \
--dataset-root=~/ColumnCW-master/MLP/dataset \
--norm=LN \
# --norm-cfg=T=1,num_groups=2,dim=2 \
--seed=1 \
--log-suffix=base \
$@
#--vis \
