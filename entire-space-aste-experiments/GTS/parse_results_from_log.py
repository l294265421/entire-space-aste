import os

from gts.common import common_path


def parse_metrics_line(line):
    """

    """
    result = {}
    parts = line.split('\t')
    result['p'] = '%.3f' % (float(parts[1][2:].strip()) * 100)
    result['r'] = '%.3f' % (float(parts[2][2:].strip()) * 100)
    result['f1'] = '%.3f' % (float(parts[3][3:].strip()) * 100)
    return result


def parse_log(filename):
    """

    """
    result = {}
    filepath = os.path.join(common_path.project_dir, 'log_bert', filename)
    with open(filepath) as input_file:
        lines = []
        target = False
        for line in input_file:
            if 'Evaluation on testset:' in line:
                target = True

            if target:
                lines.append(line)

        entire_space = True
        for line in lines:
            if 'entire_space: True' in line:
                entire_space = True
            elif 'entire_space: False' in line:
                entire_space = False
            elif 'pair' in line or 'triplet' in line:
                metrics = parse_metrics_line(line)
                result[entire_space] = metrics

    return result


models = ['bert']
data_types = ['asote.entire_space', 'asote.sentence_with_pairs']
tasks = ['triplet', 'pair']
datasets = ['res14', 'lap14', 'res15', 'res16']
for dataset in datasets:
    print(dataset)
    for model in models:
        print(model)
        for data_type in data_types:
            print(data_type)
            data_type_metrics = {}
            for task in tasks:
                task_metrics = []
                for i in range(5):
                    log_filename = '{dataset}.{model}.{task}.{i}.{data_type}.log'\
                        .format(dataset=dataset, model=model, task=task, i=i,
                                data_type=data_type)

                    metrics = parse_log(log_filename)
                    task_metrics.append(metrics)
                data_type_metrics[task] = task_metrics
            sorted_data_type_metrics = []
            sorted_data_type_metrics.append([e[True] for e in data_type_metrics['triplet']])
            sorted_data_type_metrics.append([e[True] for e in data_type_metrics['pair']])
            sorted_data_type_metrics.append([e[False] for e in data_type_metrics['triplet']])
            sorted_data_type_metrics.append([e[False] for e in data_type_metrics['pair']])
            for e in sorted_data_type_metrics:
                ps = ','.join([str(ee['p']) for ee in e])
                rs = ','.join([str(ee['r']) for ee in e])
                f1s = ','.join([str(ee['f1']) for ee in e])
                print('%s\t%s\t%s' % (ps, rs, f1s))
