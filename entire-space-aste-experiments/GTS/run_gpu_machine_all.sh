#!/usr/bin/env bash
start_time=`date +%Y%m%d%H%M%S`
echo "start ${start_time}--------------------------------------------------"

python=/root/anaconda3/bin/python

export LANG="zh_CN.UTF-8"

CUR_DIR=`pwd`

ROOT=${CUR_DIR}

export PYTHONPATH=${ROOT}:${PYTHONPATH}
# nohup sh run_gpu_machine_all.sh > run_gpu_machine_all.log 2>&1 &
datasets=(res14 lap14 res15 res16)
for dataset in ${datasets[@]}
do
  for i in `seq 0 4`
  do
    ${python} gts/NNModel/main.py --task triplet --mode train --model bilstm --dataset ${dataset} --current_run $i --data_type '.asote.entire_space' > ${dataset}.bilstm.triplet.$i.asote.entire_space.log 2>&1 &

    ${python} gts/NNModel/main.py --task triplet --mode train --model bilstm --dataset ${dataset} --current_run $i --data_type '.asote.sentence_with_pairs' > ${dataset}.bilstm.triplet.$i.asote.sentence_with_pairs.log 2>&1 &

    wait

    ${python} gts/NNModel/main.py --task pair --mode train --model bilstm --dataset ${dataset} --current_run $i --data_type '.asote.entire_space' > ${dataset}.bilstm.pair.$i.asote.entire_space.log 2>&1 &

    ${python} gts/NNModel/main.py --task pair --mode train --model bilstm --dataset ${dataset} --current_run $i --data_type '.asote.sentence_with_pairs' > ${dataset}.bilstm.pair.$i.asote.sentence_with_pairs.log 2>&1 &

    wait

  ${python} gts/NNModel/main.py --task triplet --mode train --model cnn --dataset ${dataset} --current_run $i --data_type '.asote.entire_space' > ${dataset}.cnn.triplet.$i.asote.entire_space.log 2>&1 &

    ${python} gts/NNModel/main.py --task triplet --mode train --model cnn --dataset ${dataset} --current_run $i --data_type '.asote.sentence_with_pairs' > ${dataset}.cnn.triplet.$i.asote.sentence_with_pairs.log 2>&1 &

    wait

    ${python} gts/NNModel/main.py --task pair --mode train --model cnn --dataset ${dataset} --current_run $i --data_type '.asote.entire_space' > ${dataset}.cnn.pair.$i.asote.entire_space.log 2>&1 &

    ${python} gts/NNModel/main.py --task pair --mode train --model cnn --dataset ${dataset} --current_run $i --data_type '.asote.sentence_with_pairs' > ${dataset}.cnn.pair.$i.asote.sentence_with_pairs.log 2>&1 &
    wait
  done
done

end_time=`date +%Y%m%d%H%M%S`
echo ${end_time}
