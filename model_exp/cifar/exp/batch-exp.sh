#!/bin/bash
batch_sizes=128
methods=(GN)
m_perGroup=(2 4 6 8 16 32 64 128)
lrs=(0.01 0.03 0.05 0.1)
seeds=1
T=5
NormMM=0.1
#NormMM=0.1
#affine="False"
affine="True"
width=256
arc="resLinearModel"
depth=(1 2 4 6 8 10 12 14)
epochs=150
oo="sgd"
momentum=0
wd=0
lrmethod="step"
lrstep=20
lrgamma=0.5
datasetroot="~/ccw/dataset"
#datasetroot="/home/ubuntu/leihuang/pytorch_work/data/"

l=${#batch_sizes[@]}
n=${#methods[@]}
m=${#m_perGroup[@]}
t=${#lrs[@]}
f=${#depth[@]}

for ((a=0;a<$l;++a))
do 
   for ((i=0;i<$n;++i))
   do 
      for ((j=0;j<$m;++j))
      do
        for ((k=0;k<$t;++k))
        do
          for ((b=0;b<$f;++b))
          do
                baseString="execute_${arc}_d${depth[$b]}_w${width}_b${batch_sizes[$a]}_${methods[$i]}_G${m_perGroup[$j]}_T${T}_MM${NormMM}_${affine}_lr${lrs[$k]}_s${seeds}_O${oo}"
                fileName="${baseString}.sh"
   	            echo "${baseString}"
                touch "${fileName}"
                echo  "#!/usr/bin/env bash
cd \"\$(dirname \$0)/..\" 
python3 cifar10-Randomlabel.py \\
-a=${arc} \\
--width=${width} \\
--batch-size=${batch_sizes[$a]} \\
--depth=${depth[$b]} \\
--epochs=${epochs} \\
-oo=${oo} \\
-oc=momentum=${momentum} \\
-wd=${wd} \\
--lr=${lrs[$k]} \\
--lr-method=${lrmethod} \\
--lr-step=${lrstep} \\
--lr-gamma=${lrgamma} \\
--dataset-root=${datasetroot} \\
--norm=${methods[$i]} \\
--norm-cfg=T=${T},num_groups=${m_perGroup[$j]},momentum=${NormMM},affine=${affine},dim=2 \\
--seed=${seeds} \\
--vis \\" >> ${fileName}
                echo  "nohup bash ${fileName} >output_${baseString}.out 2>&1 &" >> z_bash_excute.sh
           done
           echo  "wait" >> z_bash_excute.sh
         done
      done
   done
done
