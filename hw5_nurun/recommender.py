import os
import sys
import numpy as np

from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import cross_validate
# from surprise import evaluate, print_perf

def main():
    file_path = os.path.expanduser('ratings_small.csv')
    print(file_path)
    reader = Reader(line_format='user item rating timestamp', sep=',' ,skip_lines=1)
    data = Dataset.load_from_file(file_path, reader=reader)

    # PMF
    # To get the PMF; According to https://surprise.readthedocs.io/en/stable/matrix_factorization.html#unbiased-note
    algo = SVD(biased=False)    
    cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    # User-based collaborative filtering
    algo = KNNBasic(sim_options = {
        'user_based': True 
        })
    cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    # Item-based collaborative filtering
    algo = KNNBasic(sim_options = {
        'user_based': False 
        })
    cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    # Effect of Cosine similarity on User-based collaborative filtering
    algo = KNNBasic(sim_options = {
        'user_based': True,
        'name': 'cosine'
        })
    cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    # Effect of MSD similarity on User-based collaborative filtering
    algo = KNNBasic(sim_options = {
        'user_based': True,
        'name': 'MSD'
        })
    cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    # Effect of Pearson similarity on User-based collaborative filtering
    algo = KNNBasic(sim_options = {
        'user_based': True,
        'name': 'pearson'
        })
    cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    # Effect of Cosine similarity on Item-based collaborative filtering
    algo = KNNBasic(sim_options = {
        'user_based': False,
        'name': 'cosine'
        })
    cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    # Effect of MSD similarity on Item-based collaborative filtering
    algo = KNNBasic(sim_options = {
        'user_based': False,
        'name': 'MSD'
        })
    cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    # Effect of Pearson similarity on Item-based collaborative filtering
    algo = KNNBasic(sim_options = {
        'user_based': False,
        'name': 'pearson'
        })
    cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    # Impact of number of neighbors on User-based CF
    user_based_mean_maes = []
    user_based_mean_rmses = []
    for i in range(1, 31):
        algo = KNNBasic(k=i, sim_options = {
            'user_based': True,
            'name': 'MSD'
            })
        result = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
        user_based_mean_maes.append(np.mean(result['test_mae']))
        user_based_mean_rmses.append(np.mean(result['test_rmse']))

    print('User-based maes')
    print(' '.join(list(map(lambda x: str(round(x, 4)), user_based_mean_maes))))
    print('')
    print('User-based rmses')
    print(' '.join(list(map(lambda x: str(round(x, 4)), user_based_mean_rmses))))

    # Impact of number of neighbors on Item-based CF
    item_based_mean_maes = []
    item_based_mean_rmses = []
    for i in range(1, 31):
        algo = KNNBasic(k=i, sim_options = {
            'user_based': False,
            'name': 'MSD'
            })
        result = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
        item_based_mean_maes.append(np.mean(result['test_mae']))
        item_based_mean_rmses.append(np.mean(result['test_rmse']))

    print('Item-based maes')
    print(' '.join(list(map(lambda x: str(round(x, 4)), item_based_mean_maes))))
    print('')
    print('Item-based rmses')
    print(' '.join(list(map(lambda x: str(round(x, 4)), item_based_mean_rmses))))

if __name__ == '__main__':
    main()
