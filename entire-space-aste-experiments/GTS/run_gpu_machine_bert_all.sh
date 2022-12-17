#!/usr/bin/env bash
start_time=`date +%Y%m%d%H%M%S`
echo "start ${start_time}--------------------------------------------------"

python=/data/miniconda3/bin/python

export LANG="zh_CN.UTF-8"

CUR_DIR=`pwd`

ROOT=${CUR_DIR}

export PYTHONPATH=${ROOT}:${PYTHONPATH}
datasets=(res14 lap14 res15 res16)
for dataset in ${datasets[@]}
do
  for i in `seq 0 4`
  do
    ${python} gts/BertModel/main.py --task triplet --mode train --dataset ${dataset} --current_run $i --data_type '.asote.entire_space' > ${dataset}.bert.triplet.$i.asote.entire_space.log 2>&1 &
    wait

    ${python} gts/BertModel/main.py --task triplet --mode train --dataset ${dataset} --current_run $i --data_type '.asote.sentence_with_pairs' > ${dataset}.bert.triplet.$i.asote.sentence_with_pairs.log 2>&1 &
    wait

    ${python} gts/BertModel/main.py --task pair --mode train --dataset ${dataset} --current_run $i --data_type '.asote.entire_space' > ${dataset}.bert.pair.$i.asote.entire_space.log 2>&1 &
    wait

    ${python} gts/BertModel/main.py --task pair --mode train --dataset ${dataset} --current_run $i --data_type '.asote.sentence_with_pairs' > ${dataset}.bert.pair.$i.asote.sentence_with_pairs.log 2>&1 &
    wait
  done
done

end_time=`date +%Y%m%d%H%M%S`
echo ${end_time}
