#coding utf-8

import json, os
import random
import argparse

import torch
import torch.nn.functional as F
from tqdm import trange

from gts.BertModel.data import load_data_instances, DataIterator
from gts.BertModel.model import MultiInferBert
from gts.BertModel import utils
from gts.common import common_path


def get_model_path(args):
    """

    :param args:
    :return:
    """
    return os.path.join(args.model_dir,
                        '%s-%s-%s-%s-%s.pt' % (args.dataset, args.model, args.task, args.current_run,
                                               args.data_type))



def train(args):

    # load dataset
    train_sentence_packs = json.load(open(args.prefix + args.dataset + ('/train%s.json' % args.data_type)))
    random.shuffle(train_sentence_packs)
    dev_sentence_packs = json.load(open(args.prefix + args.dataset + ('/dev%s.json' % args.data_type)))
    instances_train = load_data_instances(train_sentence_packs, args)
    instances_dev = load_data_instances(dev_sentence_packs, args)
    random.shuffle(instances_train)
    trainset = DataIterator(instances_train, args)
    devset = DataIterator(instances_dev, args)

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    model = MultiInferBert(args).to(args.device)

    optimizer = torch.optim.Adam([
        {'params': model.bert.parameters(), 'lr': 5e-5},
        {'params': model.cls_linear.parameters()}
    ], lr=5e-5)

    best_joint_f1 = 0
    best_joint_epoch = 0
    for i in range(args.epochs):
        print('Epoch:{}'.format(i))
        for j in trange(trainset.batch_count):
            _, tokens, lengths, masks, _, _, aspect_tags, tags, tuple_flags = trainset.get_batch(j)
            preds = model(tokens, masks)

            preds_flatten = preds.reshape([-1, preds.shape[3]])
            tags_flatten = tags.reshape([-1])
            loss = F.cross_entropy(preds_flatten, tags_flatten, ignore_index=-1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        joint_precision, joint_recall, joint_f1 = eval(model, devset, args)

        if joint_f1 > best_joint_f1:
            # model_path = args.model_dir + 'bert' + args.task + '.pt'
            model_path = get_model_path(args)
            torch.save(model, model_path)
            best_joint_f1 = joint_f1
            best_joint_epoch = i
    print('best epoch: {}\tbest dev {} f1: {:.5f}\n\n'.format(best_joint_epoch, args.task, best_joint_f1))


def eval(model, dataset, args):
    result = {}
    model.eval()
    with torch.no_grad():
        all_ids = []
        all_preds = []
        all_labels = []
        all_lengths = []
        all_sens_lengths = []
        all_token_ranges = []
        all_tuple_flags = []
        for i in range(dataset.batch_count):
            sentence_ids, tokens, lengths, masks, sens_lens, token_ranges, aspect_tags, tags, tuple_flags = dataset.get_batch(i)
            preds = model(tokens, masks)
            preds = torch.argmax(preds, dim=3)
            all_preds.append(preds)
            all_labels.append(tags)
            all_lengths.append(lengths)
            all_sens_lengths.extend(sens_lens)
            all_token_ranges.extend(token_ranges)
            all_ids.extend(sentence_ids)
            all_tuple_flags.extend(tuple_flags)

        all_preds = torch.cat(all_preds, dim=0).cpu().tolist()
        all_labels = torch.cat(all_labels, dim=0).cpu().tolist()
        all_lengths = torch.cat(all_lengths, dim=0).cpu().tolist()

        for entire_space in [True, False]:
            print('entire_space: %s' % str(entire_space))
            metric = utils.Metric(args, all_preds, all_labels, all_lengths, all_sens_lengths, all_token_ranges,
                                  all_tuple_flags, ignore_index=-1, entire_space=entire_space)
            precision, recall, f1 = metric.score_uniontags()
            print(args.task + '\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}\n'.format(precision, recall, f1))

            if entire_space:
                result['.asote.entire_space'] = (precision, recall, f1)
            else:
                result['.asote.sentence_with_pairs'] = (precision, recall, f1)

            aspect_results = metric.score_aspect()
            opinion_results = metric.score_opinion()
            print('Aspect term\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(aspect_results[0], aspect_results[1],
                                                                      aspect_results[2]))
            print('Opinion term\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(opinion_results[0], opinion_results[1],
                                                                       opinion_results[2]))

    model.train()
    return result[args.data_type]


def test(args):
    print("Evaluation on testset:")
    # model_path = args.model_dir + 'bert' + args.task + '.pt'
    model_path = get_model_path(args)
    model = torch.load(model_path).to(args.device)
    model.eval()

    sentence_packs = json.load(open(args.prefix + args.dataset + ('/test%s.json' % args.data_type)))
    instances = load_data_instances(sentence_packs, args)
    testset = DataIterator(instances, args)
    eval(model, testset, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--prefix', type=str, default=common_path.common_data_dir,
                        help='dataset and embedding path prefix')
    parser.add_argument('--model_dir', type=str, default=common_path.bert_model_dir,
                        help='model path prefix')
    parser.add_argument('--task', type=str, default="pair", choices=["pair", "triplet"],
                        help='option: pair, triplet')
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"],
                        help='option: train, test')
    parser.add_argument('--model', type=str, default="bert")
    parser.add_argument('--dataset', type=str, default="res14", choices=["res14", "lap14", "res15", "res16"],
                        help='dataset')
    parser.add_argument('--max_sequence_len', type=int, default=100,
                        help='max length of a sentence')
    parser.add_argument('--device', type=str, default="cuda",
                        help='gpu or cpu')

    parser.add_argument('--bert_model_path', type=str,
                        default=common_path.project_dir + "/pretrained/bert-base-uncased",
                        help='pretrained bert model path')
    parser.add_argument('--bert_tokenizer_path', type=str,
                        default=common_path.project_dir + "/pretrained/bert-base-uncased/bert-base-uncased-vocab.txt",
                        help='pretrained bert tokenizer path')
    parser.add_argument('--bert_feature_dim', type=int, default=768,
                        help='dimension of pretrained bert feature')

    parser.add_argument('--nhops', type=int, default=1,
                        help='inference times')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='bathc size')
    parser.add_argument('--epochs', type=int, default=25,
                        help='training epoch number')
    parser.add_argument('--class_num', type=int, default=4,
                        help='label number')
    parser.add_argument('--data_type', type=str, default='',
                        choices=['', '.asote.entire_space', '.asote.sentence_with_pairs'])
    parser.add_argument('--current_run', type=int, default=0)
    args = parser.parse_args()

    if args.task == 'triplet':
        args.class_num = 6

    if args.mode == 'train':
        train(args)
        test(args)
    else:
        test(args)
