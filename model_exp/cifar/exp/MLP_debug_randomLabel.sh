#!/usr/bin/env bash
cd "$(dirname $0)/.."
python3 cifar10-mlp.py \
-a=MLP \
--width=512 \
--depth=4 \
--batch-size=256 \
--epochs=150 \
-oo=sgd \
-oc=momentum=0 \
-wd=0 \
--lr=0.9 \
--lr-method=step \
--lr-step=1 \
--lr-gamma=0.9 \
--dataset-root=~/ColumnCW-master/MLP/dataset \
--norm=LN \
# --norm-cfg=T=1,num_groups=2,dim=2 \
--seed=1 \
--log-suffix=base \
$@
#--vis \
