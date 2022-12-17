# -*- coding: utf-8 -*-


import argparse
import sys
import random
import copy
from typing import List
import json
from collections import defaultdict
import os

import torch
import numpy

from common import common_path


def parse_result_file(filepath):
    """

    :param filepath:
    :return:
    """
    result = {}
    with open(filepath) as file:
        for line in file:
            d = json.loads(line)
            sentence = ' '.join(d['tokens'])
            triplets = ['{t_start}_{t_end}_{o_start}_{o_end}_{label}'.format_map(triplet) for triplet in d['triples']
                        if triplet['label'] != '-']
            result[sentence] = triplets
    return result


def get_metrics(true_num, pred_num, tp):
    """

    :param true_num:
    :param pred_num:
    :param tp:
    :return:
    """
    precision = tp / pred_num
    recall = tp / true_num
    f1 = 2 * precision * recall / (precision + recall)
    return {'precision': '%.3f' % (precision * 100), 'recall': '%.3f' % (recall * 100), 'f1': '%.3f' % (f1 * 100)}


def remove_sentiment(triplets: List[str]):
    result = []
    for e in triplets:
        if '_-' in e:
            print()
        e = e.replace('_POS', '')
        e = e.replace('_NEG', '')
        e = e.replace('_NEU', '')
        e = e.replace('_-', '')
        result.append(e)
    return result


def evaluate_ao_pair(sentences_true, sentences_pred):
    """

    :param sentences_true:
    :param sentences_pred:
    :return:
    """
    true_triplet_num = 0
    pred_triplet_num = 0
    tp = 0
    for sentence in sentences_true.keys():
        triplets_true_temp = sentences_true[sentence]
        triplets_pred_temp = sentences_pred[sentence]

        triplets_true = remove_sentiment(triplets_true_temp)
        triplets_pred = remove_sentiment(triplets_pred_temp)

        true_triplet_num += len(triplets_true)
        pred_triplet_num += len(triplets_pred)

        for e in triplets_true:
            if e in triplets_pred:
                tp += 1
    result = get_metrics(true_triplet_num, pred_triplet_num, tp)
    return result


def evaluate_asote(sentences_true, sentences_pred):
    """

    :param sentences_true:
    :param sentences_pred:
    :return:
    """
    true_triplet_num = 0
    pred_triplet_num = 0
    tp = 0
    for sentence in sentences_true.keys():
        triplets_true = sentences_true[sentence]
        triplets_pred = sentences_pred[sentence]

        true_triplet_num += len(triplets_true)
        pred_triplet_num += len(triplets_pred)

        for e in triplets_true:
            if e in triplets_pred:
                tp += 1
    result = get_metrics(true_triplet_num, pred_triplet_num, tp)
    return result


def print_precision_recall_f1(metrics_of_multi_runs, description: str = ''):
    """

    :param metrics_of_multi_runs:
    :param description:
    :return:
    """
    # print(description)
    precisions = []
    recalls = []
    f1s = []
    for e in metrics_of_multi_runs:
        precisions.append(e['precision'])
        recalls.append(e['recall'])
        f1s.append(e['f1'])
    # print('precision: %s' % ','.join(precisions))
    # print('recall: %s' % ','.join(recalls))
    # print('f1: %s' % ','.join(f1s))
    print('%s\t%s\t%s' % (','.join(precisions), ','.join(recalls), ','.join(f1s)))


parser = argparse.ArgumentParser()
parser.add_argument('--dir_name', default='Span-ASTE-private-sentence_with_pairs-bert', type=str)
args = parser.parse_args()

configuration = args.__dict__

dir_name = configuration['dir_name']
base_dir = os.path.join(common_path.project_dir, dir_name)
result_dirs = os.listdir(base_dir)

dataset_names = ['14res', '14lap', '15res', '16res']
for dataset_name in dataset_names:
    print('dataset name: %s' % dataset_name)
    gold_types = ['entire_space', 'sentence_with_pairs']
    for gold_type in gold_types:
        # print('gold_type: %s' % gold_type)
        asote_metrics_of_multi_runs = []
        aote_metrics_of_multi_runs = []
        target_result_dirs = [os.path.join(base_dir, e) for e in result_dirs if dataset_name in e]
        for target_result_dir in target_result_dirs:
            gold_path = os.path.join(target_result_dir, 'sentences_test_gold.json')
            pred_path = os.path.join(target_result_dir, 'sentences_test_pred.json')

            gold_temp = parse_result_file(gold_path)
            if gold_type == 'sentence_with_pairs':
                gold = {key: value for key, value in gold_temp.items() if len(value) > 0}
            else:
                gold = gold_temp
            pred = parse_result_file(pred_path)

            asote_metrics_of_multi_runs.append(evaluate_asote(gold, pred))
            aote_metrics_of_multi_runs.append(evaluate_ao_pair(gold, pred))

        print_precision_recall_f1(asote_metrics_of_multi_runs, 'asote_metrics_of_multi_runs')
        print_precision_recall_f1(aote_metrics_of_multi_runs, 'ao_pair_metrics_of_multi_runs')
