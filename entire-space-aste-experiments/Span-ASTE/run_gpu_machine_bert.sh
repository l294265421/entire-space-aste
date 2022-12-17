#!/usr/bin/env bash
start_time=`date +%Y%m%d%H%M%S`
echo "start ${start_time}--------------------------------------------------"

python=/root/anaconda3/bin/python

export LANG="zh_CN.UTF-8"

CUR_DIR=`pwd`

ROOT=${CUR_DIR}

export PYTHONPATH=${ROOT}:${PYTHONPATH}

${python} aste/main.py --names 14lap,14lap,14lap,14lap,14lap,14res,14res,14res,14res,14res,15res,15res,15res,15res,15res,16res,16res,16res,16res,16res --seeds 0,1,12,123,1234,0,1,12,123,1234,0,1,12,123,1234,0,1,12,123,1234 --trainer__cuda_device 0 --trainer__num_epochs 10 --trainer__checkpointer__num_serialized_models_to_keep 1 --model__span_extractor_type "endpoint" --model__modules__relation__use_single_pool False --model__relation_head_type "proper" --model__use_span_width_embeds True --model__modules__relation__use_distance_embeds True --model__modules__relation__use_pair_feature_multiply False --model__modules__relation__use_pair_feature_maxpool False --model__modules__relation__use_pair_feature_cls False --model__modules__relation__use_span_pair_aux_task False --model__modules__relation__use_span_loss_for_pruners False --model__loss_weights__ner 1.0 --model__modules__relation__spans_per_word 0.5 --model__modules__relation__neg_class_weight -1


end_time=`date +%Y%m%d%H%M%S`
echo ${end_time}
