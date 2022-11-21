import numpy as np
import time
import scipy.sparse as sparse
from collections import defaultdict

from lib.PoissonFactorModel import PoissonFactorModel
from lib.MultiGaussianModel import MultiGaussianModel
from lib.TimeAwareMF import TimeAwareMF
from lib.FriendBasedCF import FriendBasedCF
from lib.metrics import precisionk, recallk, ndcgk, mapk
from lib.LocationFriendshipBookmarkColoringAlgorithm import LocationFriendshipBookmarkColoringAlgorithm

def read_friend_data():
    social_data = open(social_file, 'r').readlines()
    social_matrix = np.zeros((user_num, user_num))
    for eachline in social_data:
        uid1, uid2 = eachline.strip().split()
        uid1, uid2 = int(uid1), int(uid2)
        social_matrix[uid1, uid2] = 1.0
        social_matrix[uid2, uid1] = 1.0
    return social_matrix


def read_poi_coos():
    poi_coos = {}
    poi_data = open(poi_file, 'r').readlines()
    for eachline in poi_data:
        lid, lat, lng = eachline.strip().split()
        lid, lat, lng = int(lid), float(lat), float(lng)
        poi_coos[lid] = (lat, lng)
    return poi_coos

def read_training_data2():
    train_data = open(train_file, 'r').readlines()
    training_matrix = np.zeros((user_num, poi_num))
    for eachline in train_data:
        uid, lid, freq = eachline.strip().split()
        uid, lid, freq = int(uid), int(lid), int(freq)
        training_matrix[uid, lid] = freq
    return training_matrix


def read_training_data():
    # load train data
    train_data = open(train_file, 'r').readlines()
    sparse_training_matrix = sparse.dok_matrix((user_num, poi_num))
    training_matrix = np.zeros((user_num, poi_num))
    training_tuples = set()
    for eachline in train_data:
        uid, lid, freq = eachline.strip().split()
        uid, lid, freq = int(uid), int(lid), int(freq)
        sparse_training_matrix[uid, lid] = freq
        training_matrix[uid, lid] = 1.0
        training_tuples.add((uid, lid))

    # load checkins
    # time_list_hour = open("./result/time_hour" + ".txt", 'w')
    check_in_data = open(check_in_file, 'r').readlines()
    training_tuples_with_day = defaultdict(int)
    training_tuples_with_time = defaultdict(int)
    for eachline in check_in_data:
        uid, lid, ctime = eachline.strip().split()
        uid, lid, ctime = int(uid), int(lid), float(ctime)
        if (uid, lid) in training_tuples:
            hour = time.gmtime(ctime).tm_hour
            training_tuples_with_time[(hour, uid, lid)] += 1.0
            if 8 <= hour < 18:
                # working time
                hour = 0
            elif hour >= 18 or hour < 8:
                # leisure time
                hour = 1

            training_tuples_with_day[(hour, uid, lid)] += 1.0

    # Default setting: time is partitioned to 24 hours.
    sparse_training_matrices = [sparse.dok_matrix((user_num, poi_num)) for _ in range(24)]
    for (hour, uid, lid), freq in training_tuples_with_time.items():
        sparse_training_matrices[hour][uid, lid] = 1.0 / (1.0 + 1.0 / freq)

    # Default setting: time is partitioned to WD and WE.
    sparse_training_matrix_WT = sparse.dok_matrix((user_num, poi_num))
    sparse_training_matrix_LT = sparse.dok_matrix((user_num, poi_num))

    for (hour, uid, lid), freq in training_tuples_with_day.items():
        if hour == 0:
            sparse_training_matrix_WT[uid, lid] = freq
        elif hour == 1:
            sparse_training_matrix_LT[uid, lid] = freq

    print ("Data Loader Finished!")
    return sparse_training_matrices, sparse_training_matrix, sparse_training_matrix_WT, sparse_training_matrix_LT, training_tuples, training_matrix


def read_ground_truth():
    ground_truth = defaultdict(set)
    truth_data = open(test_file, 'r').readlines()
    for eachline in truth_data:
        uid, lid, _ = eachline.strip().split()
        uid, lid = int(uid), int(lid)
        ground_truth[uid].add(lid)
    print("The loading of Ground Truth Finished.")
    return ground_truth


def main():
    sparse_training_matrices, sparse_training_matrix, sparse_training_matrix_WT, sparse_training_matrix_LT, training_tuples, training_matrix = read_training_data()
    ground_truth = read_ground_truth()
    training_matrix2 = read_training_data2()
    poi_coos = read_poi_coos()
    # social_relations = read_friend_data()
    social_matrix = read_friend_data()

    start_time = time.time()

    PFM.train(sparse_training_matrix, max_iters=10, learning_rate=1e-4)
    PFM.save_model("./tmp/")
    PFM.load_model("./tmp/")

    ctime = time.time()

    # Multi-Center Weekday
    MGMWT.multi_center_discovering(sparse_training_matrix_WT, poi_coos)
    # Multi-Center Weekend
    MGMLT.multi_center_discovering(sparse_training_matrix_LT, poi_coos)

    print("MGM Done. Elapsed time:", time.time() - ctime, "s")

    TAMF.train(sparse_training_matrices, max_iters=30, load_sigma=False)
    TAMF.save_model("./tmp/")
    TAMF.load_model("./tmp/")

    LFBCA.precompute_rec_scores(training_matrix2, social_matrix)
    LFBCA.save_result("./tmp/")

    # S.compute_friend_sim(social_relations, training_matrix)

    elapsed_time = time.time() - start_time
    print("Done. Elapsed time:", elapsed_time, "s")

    execution_time = open("./result/execution_time" + ".txt", 'w')
    execution_time.write(str(elapsed_time))

    rec_list = open("./result/reclist_top_" + str(top_k) + ".txt", 'w')
    result_5 = open("./result/result_top_" + str(5) + ".txt", 'w')
    result_10 = open("./result/result_top_" + str(10) + ".txt", 'w')
    result_15 = open("./result/result_top_" + str(15) + ".txt", 'w')
    result_20 = open("./result/result_top_" + str(20) + ".txt", 'w')

    all_uids = list(range(user_num))
    all_lids = list(range(poi_num))
    np.random.shuffle(all_uids)

    # list for different ks
    precision_5, recall_5, nDCG_5, MAP_5 = [], [], [], []
    precision_10, recall_10, nDCG_10, MAP_10 = [], [], [], []
    precision_15, recall_15, nDCG_15, MAP_15 = [], [], [], []
    precision_20, recall_20, nDCG_20, MAP_20 = [], [], [], []

    for cnt, uid in enumerate(all_uids):
        if uid in ground_truth:
            # What is the meaning of the following structure?
            overall_scores = [PFM.predict(uid, lid) * (MGMWT.predict(uid, lid) + MGMLT.predict(uid, lid)) * TAMF.predict(uid, lid) * LFBCA.predict(uid, lid)
                              if (uid, lid) not in training_tuples else -1
                              for lid in all_lids]

            overall_scores = np.array(overall_scores)

            predicted = list(reversed(overall_scores.argsort()))[:top_k]
            actual = ground_truth[uid]


            # calculate the average of different k
            precision_5.append(precisionk(actual, predicted[:5]))
            recall_5.append(recallk(actual, predicted[:5]))
            nDCG_5.append(ndcgk(actual, predicted[:5]))
            MAP_5.append(mapk(actual, predicted[:5], 5))

            precision_10.append(precisionk(actual, predicted[:10]))
            recall_10.append(recallk(actual, predicted[:10]))
            nDCG_10.append(ndcgk(actual, predicted[:10]))
            MAP_10.append(mapk(actual, predicted[:10], 10))

            precision_15.append(precisionk(actual, predicted[:15]))
            recall_15.append(recallk(actual, predicted[:15]))
            nDCG_15.append(ndcgk(actual, predicted[:15]))
            MAP_15.append(mapk(actual, predicted[:15], 15))

            precision_20.append(precisionk(actual, predicted[:20]))
            recall_20.append(recallk(actual, predicted[:20]))
            nDCG_20.append(ndcgk(actual, predicted[:20]))
            MAP_20.append(mapk(actual, predicted[:20], 20))

            print(cnt, uid, "pre@10:", np.mean(precision_10), "rec@10:", np.mean(recall_10))

            rec_list.write('\t'.join([
                str(cnt),
                str(uid),
                ','.join([str(lid) for lid in predicted])
            ]) + '\n')

            # write the different ks
            result_5.write('\t'.join([str(cnt), str(uid), str(precision_5), str(recall_5), str(nDCG_5), str(MAP_5)]) + '\n')
            result_10.write('\t'.join([str(cnt), str(uid), str(precision_10), str(recall_10), str(nDCG_10), str(MAP_10)]) + '\n')
            result_15.write('\t'.join([str(cnt), str(uid), str(precision_15), str(recall_15), str(nDCG_15), str(MAP_15)]) + '\n')
            result_20.write('\t'.join([str(cnt), str(uid), str(precision_20), str(recall_20), str(nDCG_20), str(MAP_20)]) + '\n')
    print("<< SUCP is Finished >>")


if __name__ == '__main__':
    data_name = sys.argv[1]  # which data to run
    beta_value = sys.argv[2]  # which beta value to use


    print("======= RUNNING FOR BETA = ", beta_value, ", DATASET = ", data_name, "========")

    if data_name == 'gowalla':
        data_dir = "./Dataset/Gowalla/"

        size_file = data_dir + "Gowalla_data_size.txt"
        check_in_file = data_dir + "Gowalla_checkins.txt"
        train_file = data_dir + "Gowalla_train.txt"
        tune_file = data_dir + "Gowalla_tune.txt"
        test_file = data_dir + "Gowalla_test.txt"
        poi_file = data_dir + "Gowalla_poi_coos.txt"
        social_file = data_dir + "Gowalla_social_relations.txt"
    else:
        data_dir = "./Dataset/Yelp/"

        size_file = data_dir + "Yelp_data_size.txt"
        check_in_file = data_dir + "Yelp_checkins.txt"
        train_file = data_dir + "Yelp_train.txt"
        tune_file = data_dir + "Yelp_tune.txt"
        test_file = data_dir + "Yelp_test.txt"
        poi_file = data_dir + "Yelp_poi_coos.txt"
        social_file = data_dir + "Yelp_social_relations.txt"

    user_num, poi_num = open(size_file, 'r').readlines()[0].strip('\n').split()
    user_num, poi_num = int(user_num), int(poi_num)

    top_k = 100

    PFM = PoissonFactorModel(K=30, alpha=20.0, beta=0.2)
    MGMWT = MultiGaussianModel(alpha=0.2, theta=0.02, dmax=15)
    MGMLT = MultiGaussianModel(alpha=0.2, theta=0.02, dmax=15)
    TAMF = TimeAwareMF(K=100, Lambda=1.0, beta=2.0, alpha=2.0, T=24)
    LFBCA = LocationFriendshipBookmarkColoringAlgorithm(alpha=0.85, beta=float(beta_value), epsilon=0.001)

    tmp_dir_name = "./tmp_{}_{}/".format(data_name, beta_value)
    result_dir_name = "./result_{}_{}/".format(data_name, beta_value)
    try:
        os.makedirs(tmp_dir_name)
    except OSError as e:
        pass

    try:
        os.makedirs(result_dir_name)
    except OSError as e:
        pass

    main(result_dir_name, tmp_dir_name)
