import pickle
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from gan_attack_implement import get_filename
def read_pkl_origin(filename, mode, appendix):
    np.set_printoptions(threshold=np.inf)   # 解决显示不完全问题

    # filename = "/home/shengy/luoshenseeker/Labs-Federated-Learning/data/NIID-Bench-origin/saved_exp_info/acc/" + filename
    # print('!origin!')
    filename = f"/home/shengy/luoshenseeker/AIJack/output/{mode}/" + filename
    print('!old!')

    fr=open(filename,'rb')

    acc_hist = pickle.load(fr)


    # 新版需用
    server_acc = acc_hist
    # server_loss = np.dot(weights, loss_hist[i + 1])
    # for i in range(len(acc_hist)):
        # n_samples = np.array([200 for _ in range(2)])
        # weights = n_samples / np.sum(n_samples)   # sample size为聚合权重
        # if np.dot(weights, acc_hist[i]) != 0.0:
        #     server_acc.append(np.dot(weights, acc_hist[i]))
    if appendix == "mr":
        return server_acc[:, 0]
    else:
        return server_acc[:, 1]

def plot_acc_with_order(exp_name: str, pkl_file: list, legends = [], mode="acc", line_type="smooth", appendix="mr"):
    plt.clf()
    # '1.3', 2.6.c
    # exp_name = ['1.1', '1.2', '1.4', '1.5', '1.6', '1.7', '2.1', '2.2', '2.3', '2.4', '2.5', '2.7']
    # exp_name = ['4.0']
    # print(exp_name_)
    # pkl_file = not_final_pkl_dict[exp_name_]
    # pkl_file = [
    #     "MNIST_iid_FedAvg_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_AVG_conv.pkl",
    #     "MNIST_iid_random_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_AVG_conv.pkl",
    #     "MNIST_iid_clustered_1_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_AVG_conv.pkl",
    #     "MNIST_iid_ours_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_AVG_conv.pkl",
    #     # "MNIST_iid_random_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_SCAFFOLD_conv.pkl",
    #     # "MNIST_iid_random_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_SCAFFOLD_ncon.pkl"
    # ]

    # exp_name = get_exp_name(pkl_file[0])

    if pkl_file[0][:5] == "MNIST" or pkl_file[0][:6] == "FMNIST":
        start = 0
        end = 200
        op = 5
        stp = 5
    elif pkl_file[0][:5] == "CIFAR":
        start = 0
        end = 800
        op = 10
        stp = 10
    else:
        start = 0
        end = 200
        op = 5
        stp = 5

    y = {}
    meanop = {}
    meanop_5 = {}
    colors = ['#ff851b','#3d6daa','#c04851','#806d9e','#66a9c9',   '#ff851b','#3d6daa','#c04851','#806d9e','#66a9c9']  # 黄 浅蓝 红 紫 深蓝 alg1
    # colors = ['#ff851b','#806d9e','#c04851','#3d6daa','#66a9c9']  # 黄 紫 红 浅蓝 深蓝 md

    acc = pd.DataFrame()

    n = len(pkl_file)
    for k in range(n):

        # if pkl_file[k][:1] == 'm' or pkl_file[k][:1] == 'f' or pkl_file[k][:1] == 'c':
        acc[k] = read_pkl_origin(pkl_file[k], mode=mode, appendix=appendix)
        # else:
        #     y[k] = read_pkl(pkl_file[k])
        # y[k] = pkl_file[k]
        # print(y[k])
        # acc[k][0] = y[k][0][start:end]
        # acc[k][1] = y[k][1][start:end]
        print(round(acc[k], 2).tolist())
        if end > len(acc[k]):
            end = len(acc[k])
        plt.plot(range(start+1, end+1), acc[k])
    # xxx = [0 for _ in range(len(pkl_file))]
    # for x in range(len(pkl_file)):
    #     xxx[x] = pkl_file[x][-9:-4]
    # plt.legend(labels=xxx)

    if not legends:
        if n == 4:
            plt.legend(labels=["Random", "Importance", "Cluster", "Ours"])
        elif n == 3:
            # plt.legend(loc='lower right', labels=["random_sampling", "cluster_sampling", "ours"])
            plt.legend(loc='lower right',labels=["random_sampling", "importance_sampling", "ours"])
            # plt.legend(loc='lower right',labels=["random_sampling", "importance_sampling", "ours"])
    else:
        plt.legend(labels=legends)

    # filename = exp_name+'_acc'
    plt.xlim([start, end])  #设置x轴显示的范围
    plt.grid()
    plt.xlabel('Communication Rounds', {'size':15})
    plt.ylabel('Test Accuracy', {'size':15})
    plt.title(exp_name, {'size':18})
    # # plt.title('MNIST Non-iid p=1', {'size':18})  # title的大小设置为18
    plt.savefig(f'/home/shengy/luoshenseeker/AIJack/output/plot_result/{exp_name}_{mode}_{appendix}.png', format='png', dpi=600, bbox_inches="tight")
    # plt.show()

    print("Saved")
    # plt.clf()

if __name__ == "__main__":
    # plot_acc_with_order("Mnist shard conv", [
    #     "MNIST_shard_FedAvg_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_AVG_conv.pkl",
    #     "MNIST_shard_random_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_AVG_conv.pkl", 
    #     "MNIST_shard_clustered_1_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_AVG_conv.pkl",
    #     "MNIST_shard_ours_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_AVG_conv.pkl",
    #     ])
    file_list = [
        "normal_gepoch5_pp.pkl",
        "dp_n1_d1.0_e200.0_pp.pkl",
        # "dp_n1_d1.0_e100.0_np.pkl",
        # "dp_n1_d1.0_e20.0_np.pkl",
        # "dp_n1_d1.0_e10.0_np.pkl",
        # "dp_n1_d1.0_e1.0_np.pkl"
    ]
    legends = ["normal", "e=200", "e=100", "e=20", "e=10", "e=1"]
    exp_name = "mnist_compare_pp"
    plot_acc_with_order(exp_name, file_list,
        legends,
        mode="similarity_score",
        line_type="straight",
        appendix="mr"
    )
    plot_acc_with_order(exp_name, file_list,
        legends,
        mode="similarity_score",
        line_type="straight",
        appendix="ssim"
    )
    # plot_acc_with_order(exp_name, file_list,
    #     legends,
    #     mode="loss"
    # )

    # 0.5 touch 60: 
