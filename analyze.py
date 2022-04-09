import numpy as np

seed_list = [31, 37, 41, 42, 43, 47, 53]
type_list = ['tumor', 'T1SS', 'T2SS', 'T1', '2label', 'randT1', 'randTumor']


def get_test_result(test_scores):
    print(f'{np.round(np.mean(test_scores, axis=0), decimals=4)}')


for type in type_list:
    for epochs in range(10, 61, 10):
        all_scores = []
        for seed in seed_list:
            directory = f'../output_V100/seed{seed}/{type}/{type}_seed{seed}-ep{epochs}/'
            filename = 'train.log'
            with open(directory + filename, 'r') as f:
                line = f.readlines()[-3]
                l = line.find('[')
                r = line.find(']')
                scores = []
                for score in line[l + 1: r].strip().split(' '):
                    if score != '':
                        scores.append(score)
                # print(scores)
                all_scores.append(np.array(scores).astype('float64'))
        print(f'{type} {epochs}: ', end='')
        get_test_result(np.array(all_scores))
    print()

