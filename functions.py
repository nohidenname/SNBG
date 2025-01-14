import nibabel as nib
import numpy as np
import networkx as nx
import os
import pandas as pd
from sklearn import svm
from sklearn.model_selection import StratifiedKFold, cross_val_score

def get_matrices(csv_folder_path,csv_files):
    matrices = []
    for csv_file in csv_files:
        file_path = os.path.join(csv_folder_path, csv_file)
        df = pd.read_csv(file_path)
        matrix = df.to_numpy() 
        matrices.append(matrix)
    return matrices

def get_sub_list(matrices,df):
    sub_list = []
    for matrix in matrices:
        brain_data = matrix
        # 创建节点编号到网络的映射
        node_network_mapping = df.set_index('Node')['Network'].to_dict()

        # 获取所有唯一的网络
        unique_networks = np.unique(df['Network'])

        # 创建一个列表，用于存储每个网络的数据矩阵
        network_matrices = []

        # 将脑区数据按网络存储
        for network in unique_networks:
            node_ids = df[df['Network'] == network]['Node'].tolist()
            network_data_matrix = brain_data[:, np.array(node_ids)-1]
            # 将每个网络内的数据转换为数组，并加入列表
            network_matrices.append(network_data_matrix)

        # 将当前窗口的组数据存储到列表中
        sub_list.append(network_matrices)
    return sub_list

def get_groups(sub_list):
    group_MF = []
    group_FP = []
    group_DM = []
    for sub_idx,sub_data in enumerate(sub_list,start=1):
        for network_idx,network_data_matrix in enumerate(sub_data,start=1):
            network_number = f"group{network_idx}"
            if network_number == 'group1':
                group_MF.append(network_data_matrix)
            elif network_number == 'group2':
                group_FP.append(network_data_matrix)
            elif network_number == 'group3':
                group_DM.append(network_data_matrix)
    return group_MF,group_FP,group_DM

def get_cut_data(group_data,window_size,stride):
    cut_data = []
    for matrix in group_data:
        num_windows = (matrix.shape[0] - window_size)//stride + 1
        sub_cut = []
        for i in range(0,num_windows * stride,stride):
            window = matrix[i:i + window_size,:]
            sub_cut.append(window)
        cut_data.append(sub_cut)
    return cut_data

def pcc_method(cut_data):
    threshold_list = [55,60,65,70,75,80,85,90,95]
    thresh_cut_data = []
    for threshold in threshold_list:
        cut_cor = []
        for sub_cut in cut_data:
            sub_pear = []
            for matrix in sub_cut:
                cor_matrix = np.corrcoef(matrix,rowvar=False)
                np.fill_diagonal(cor_matrix,0)
                threshold_value = np.percentile(np.abs(cor_matrix),threshold)
                binary_matrix = np.where(np.abs(cor_matrix) >= threshold_value,1,0)
                sub_pear.append(binary_matrix)
            cut_cor.append(sub_pear)
        thresh_cut_data.append(cut_cor)
    return thresh_cut_data

def snbg_cn(M,N,matrix_list):
    all_matrix = np.vstack(matrix_list)
    Ea = np.corrcoef(all_matrix,rowvar=False)  # step-1  network e(a)
    Ea = np.nan_to_num(Ea)  
    removed_list = [matrix for matrix in matrix_list if not np.array_equal(matrix, M)]
    M_q = np.vstack(removed_list)
    Ea_q = np.corrcoef(M_q,rowvar=False)  # step-2  network e(a_q)
    Ea_q = np.nan_to_num(Ea_q)
    Eq = N * (Ea - Ea_q) + Ea_q  # step-3
    return Eq

def snbg_ad(M,N,cn_matrix_list):
    new_matrix_list = cn_matrix_list + [M]
    all_matrix = np.vstack(new_matrix_list)
    Ea = np.corrcoef(all_matrix,rowvar=False)
    Ea = np.nan_to_num(Ea)
    M_q = np.vstack(cn_matrix_list)
    Ea_q = np.corrcoef(M_q,rowvar=False)
    Ea_q = np.nan_to_num(Ea_q)
    Eq = N * (Ea - Ea_q) + Ea_q
    return Eq

def thresh_list_cn(window_list):
    res_list = []
    threshold_list = [55,60,65,70,75,80,85,90,95]
    for threshold in threshold_list:
        cut_list = []
        for window in window_list:
            sub_list = []
            for matrix in window:
                cor_matrix = np.triu(snbg_cn(matrix,len(window),window))
                np.fill_diagonal(cor_matrix,0)
                threshold_val = np.percentile(np.abs(cor_matrix),threshold)
                binary_matrix = np.where(np.abs(cor_matrix) >= threshold_val,1,0)
                sub_list.append(binary_matrix)
            cut_list.append(sub_list)
        res_list.append(cut_list)
    return res_list
def thresh_list_ad(ad_window_list,cn_window_list):
    res_list = []
    threshold_list = [55,60,65,70,75,80,85,90,95]
    for threshold in threshold_list:
        cut_list = []
        for i,window in enumerate(ad_window_list):
            sub_list = []
            for matrix in window:
                cor_matrix = np.triu(snbg_ad(matrix,len(window)+1,cn_window_list[i]))
                np.fill_diagonal(cor_matrix,0)
                threshold_val = np.percentile(np.abs(cor_matrix),threshold)
                binary_matrix = np.where(np.abs(cor_matrix) >= threshold_val,1,0)
                sub_list.append(binary_matrix)
            cut_list.append(sub_list)
        res_list.append(cut_list)
    return res_list

def turn_construct(thresh_list):
    res = []
    for threshold in thresh_list:
        zipped_list = list(zip(*threshold))
        window_sub_list = [list(x) for x in zipped_list]
        res.append(window_sub_list)
    return res

def build_network_from_correlation_matrix(correlation_matrix):
    G = nx.Graph()
    num_nodes = correlation_matrix.shape[0]
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            correlation = correlation_matrix[i, j]
            G.add_edge(i, j, weight=correlation)
    return G

def calculate_degree_matrix(data_list):
    res = []
    for matrix_list in data_list:
        num_graphs = len(matrix_list)
        max_nodes = matrix_list[0].shape[0]
        degree_matrix = np.zeros((num_graphs, max_nodes))
        for i, matrix in enumerate(matrix_list):
            matrix = np.triu(matrix)
            G = build_network_from_correlation_matrix(matrix)
            edges_to_remove = [(u, v) for u, v, w in G.edges(data='weight') if w == 0]
            G.remove_edges_from(edges_to_remove)
            degree_dict = dict(G.degree())
            degrees = [degree_dict[node] for node in G.nodes()]
            degree_matrix[i,:len(degrees)] = degrees
        res.append(degree_matrix)
    return res

def degree_matrix_res(data_raw):
    res = calculate_degree_matrix(data_raw)
    degree = []
    for sub in res:
        matrices = np.array(sub)
        mean_res = np.mean(matrices,axis=0)
        degree.append(mean_res)
    degree_matrix = np.vstack((degree))
    return degree_matrix

def degree_matrix_list(matrix_list):
    res = []
    for list_data in matrix_list:
        matrix = degree_matrix_res(list_data)
        res.append(matrix)
    return res

def calculate_feature(ad_matrix,cn_matrix):
    label_add_ad = 'AD'
    label_add_cn = 'CN'
    labels_ad = np.full((ad_matrix.shape[0],1),label_add_ad,dtype='object')
    data_labels_ad = np.column_stack((ad_matrix,labels_ad))
    '''cn'''
    labels_cn = np.full((cn_matrix.shape[0],1),label_add_cn,dtype='object')
    data_labels_cn = np.column_stack((cn_matrix,labels_cn))
    feature = np.vstack((data_labels_ad,data_labels_cn))
    return feature

def get_clustering(matrix):
    processed_matrix = np.triu(matrix)
    G = build_network_from_correlation_matrix(processed_matrix)
    edges_to_remove = [(u, v) for u, v, w in G.edges(data='weight') if w == 0]
    G.remove_edges_from(edges_to_remove)
    clustering_coefficients = nx.clustering(G)
    clustering_values = list(clustering_coefficients.values())
    clustering_array = np.array(clustering_values)
    return clustering_array

def get_clustering_list(matrix_list):
    res_list = []
    for threshold in matrix_list:
        sub_clustering = []
        for sub in threshold:
            clustering_list = []
            for matrix in sub:
                clustering = get_clustering(matrix)
                clustering_list.append(clustering)
            clustering_matrix = np.vstack(clustering_list)
            feature_list = np.mean(clustering_matrix,axis=0)
            sub_clustering.append(feature_list)
        res_list.append(sub_clustering)
    return res_list

def concatenate_clustering_feature(feature_list):
    res_list = []
    for threshold in feature_list:
        res = np.vstack(threshold)
        res_list.append(res)
    feature = np.vstack(res_list)
    return feature

def libsvm(matrix, cost):
    X = matrix[:, :-1] 
    y = matrix[:, -1]   
    clf = svm.SVC(kernel='rbf', C=cost)
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    accuracies = cross_val_score(clf, X, y, cv=skf, scoring='accuracy')
    total_accuracy = np.mean(accuracies)
    std_deviation = np.std(accuracies)
    total_accuracy_percent = total_accuracy * 100
    std_deviation_percent = std_deviation * 100

    print(f' {total_accuracy_percent:.2f}% ± {std_deviation_percent:.2f}%')
    
    return accuracies, total_accuracy, std_deviation

def coefficient_of_variation(matrix):
    means = np.mean(matrix, axis=0)
    variances = np.std(matrix, axis=0)
    coefficients = variances/ means
    return coefficients

def get_rsd_list(degree_data):
    rsd_res = []
    for threshold in degree_data:
        rsd = np.round(coefficient_of_variation(threshold),4)
        rsd_res.append(rsd)
    return rsd_res

def get_combine_res(rsd_list1,rsd_list2):
    rsd_list_res = []
    for i in range(len(rsd_list1)):
        rsd_matrix = np.vstack((rsd_list1[i],rsd_list2[i]))
        rsd_list_res.append(rsd_matrix.T)
    return rsd_list_res

def global_efficiency(G):
    N = len(G.nodes)
    total_efficiency = 0
    total_pairs = 0
    for component in nx.connected_components(G):
        component_efficiency = 0
        shortest_paths = dict(nx.shortest_path_length(G.subgraph(component)))
        for i in component:
            for j in component:
                if i != j:
                    component_efficiency += 1 / shortest_paths[i][j]
        total_efficiency += component_efficiency
        total_pairs += len(component) * (len(component) - 1)
    if total_pairs == 0:
        return 0
    global_efficiency = total_efficiency / total_pairs
    return global_efficiency

def get_global_efficiency_list(matrix_list):
    res = []
    for sub in matrix_list:
        sub_res = []
        for matrix in sub:
            matrix = np.triu(matrix)
            G = build_network_from_correlation_matrix(matrix)
            edges_to_remove = [(u, v) for u, v, w in G.edges(data='weight') if w == 0]
            G.remove_edges_from(edges_to_remove)
            efficiency = global_efficiency(G)
            sub_res.append(efficiency)
        eff = np.mean(sub_res)
        res.append(eff)
    return res

def get_ec_feature(data_list):
    res = []
    for i in range(len(data_list)):
        ec_res = get_global_efficiency_list(data_list[i])
        res.append(ec_res)
    matrix = np.vstack((res))
    return matrix

def concate_matrix(matrix1,matrix2):
    ec_feature = []
    for row_index in range(matrix1.shape[0]):
        row_A = matrix1[row_index]
        row_B = matrix2[row_index]
        new_matrix = np.vstack([row_A, row_B])
        matrix = new_matrix.T
        ec_feature.append(matrix)
    return ec_feature

def get_ec_rsd_matrix(ec_list):
    ec_res = []
    for matrix in ec_list:
        rsd = coefficient_of_variation(matrix)
        ec_res.append(rsd)
    matrix = np.vstack(ec_res)
    return matrix
def get_traget_ec_matrix(matrix1,matrix2,matrix3,matrix4,matrix5,matrix6):
    matrix_list = []
    for i in range(matrix1.shape[0]):
        thresh_list = []
        thresh_list.append(matrix1[i])
        thresh_list.append(matrix2[i])
        thresh_list.append(matrix3[i])
        thresh_list.append(matrix4[i])
        thresh_list.append(matrix5[i])
        thresh_list.append(matrix6[i])
        thresh_matrix = np.vstack((thresh_list))
        matrix_list.append(thresh_matrix)
    return matrix_list

def get_test_data(ad_matrices,cn_matrices,W,s):
    #select atlas file
    df = pd.read_csv(r"your atlas file")

    ad_sub_list = get_sub_list(ad_matrices,df)
    group_ad_MF,group_ad_FP,group_ad_DM = get_groups(ad_sub_list)

    cn_sub_list = get_sub_list(cn_matrices,df)
    group_cn_MF,group_cn_FP,group_cn_DM = get_groups(cn_sub_list)

    window_size = W
    stride = s
    cut_ad_MF = get_cut_data(group_ad_MF,window_size,stride)
    cut_ad_FP = get_cut_data(group_ad_FP,window_size,stride)
    cut_ad_DM = get_cut_data(group_ad_DM,window_size,stride)
    cut_cn_MF = get_cut_data(group_cn_MF,window_size,stride)
    cut_cn_FP = get_cut_data(group_cn_FP,window_size,stride)
    cut_cn_DM = get_cut_data(group_cn_DM,window_size,stride)

    #pcc_based method
    ad_thresh_cut_MF = pcc_method(cut_ad_MF)
    ad_thresh_cut_FP = pcc_method(cut_ad_FP)
    ad_thresh_cut_DM = pcc_method(cut_ad_DM)
    cn_thresh_cut_MF = pcc_method(cut_cn_MF)
    cn_thresh_cut_FP = pcc_method(cut_cn_FP)
    cn_thresh_cut_DM = pcc_method(cut_cn_DM)

    #snbg method
    zipped_list_ad_MF = list(zip(*cut_ad_MF))
    zipped_list_ad_FP = list(zip(*cut_ad_FP))
    zipped_list_ad_DM = list(zip(*cut_ad_DM))
    zipped_list_cn_MF = list(zip(*cut_cn_MF))
    zipped_list_cn_FP = list(zip(*cut_cn_FP))
    zipped_list_cn_DM = list(zip(*cut_cn_DM))
    window_sub_ad_MF = [list(x) for x in zipped_list_ad_MF]
    window_sub_ad_FP = [list(x) for x in zipped_list_ad_FP]
    window_sub_ad_DM = [list(x) for x in zipped_list_ad_DM]
    window_sub_cn_MF = [list(x) for x in zipped_list_cn_MF]
    window_sub_cn_FP = [list(x) for x in zipped_list_cn_FP]
    window_sub_cn_DM = [list(x) for x in zipped_list_cn_DM]

    ad_thresh_MF = thresh_list_ad(window_sub_ad_MF,window_sub_cn_MF)
    ad_thresh_FP = thresh_list_ad(window_sub_ad_FP,window_sub_cn_FP)
    ad_thresh_DM = thresh_list_ad(window_sub_ad_DM,window_sub_cn_DM)
    cn_thresh_MF = thresh_list_cn(window_sub_cn_MF)
    cn_thresh_FP = thresh_list_cn(window_sub_cn_FP)
    cn_thresh_DM = thresh_list_cn(window_sub_cn_DM)


    ad_thresh_cut_MF_snbg = turn_construct(ad_thresh_MF)
    ad_thresh_cut_FP_snbg = turn_construct(ad_thresh_FP)
    ad_thresh_cut_DM_snbg = turn_construct(ad_thresh_DM)
    cn_thresh_cut_MF_snbg = turn_construct(cn_thresh_MF)
    cn_thresh_cut_FP_snbg = turn_construct(cn_thresh_FP)
    cn_thresh_cut_DM_snbg = turn_construct(cn_thresh_DM)

    return ad_thresh_cut_MF,ad_thresh_cut_FP,ad_thresh_cut_DM,cn_thresh_cut_MF,cn_thresh_cut_FP,cn_thresh_cut_DM,ad_thresh_cut_MF_snbg,ad_thresh_cut_FP_snbg,ad_thresh_cut_DM_snbg,cn_thresh_cut_MF_snbg,cn_thresh_cut_FP_snbg,cn_thresh_cut_DM_snbg

def robust_test(ad_thresh_cut_MF,ad_thresh_cut_FP,ad_thresh_cut_DM,cn_thresh_cut_MF,cn_thresh_cut_FP,cn_thresh_cut_DM,ad_thresh_cut_MF_snbg,ad_thresh_cut_FP_snbg,ad_thresh_cut_DM_snbg,cn_thresh_cut_MF_snbg,cn_thresh_cut_FP_snbg,cn_thresh_cut_DM_snbg):
    '''PCC-based'''
    #degree calculate
    ad_degree_DM = degree_matrix_list(ad_thresh_cut_DM)
    ad_degree_MF = degree_matrix_list(ad_thresh_cut_MF)
    ad_degree_FP = degree_matrix_list(ad_thresh_cut_FP)
    cn_degree_DM = degree_matrix_list(cn_thresh_cut_DM)
    cn_degree_MF = degree_matrix_list(cn_thresh_cut_MF)
    cn_degree_FP = degree_matrix_list(cn_thresh_cut_FP)
    #network's degree features
    list_degree_DM = []
    list_degree_FP = []
    list_degree_MF = []
    degree_threshold_list = []

    for i in range(len(ad_degree_DM)):
        feature_DM = calculate_feature(ad_degree_DM[i],cn_degree_DM[i])
        list_degree_DM.append(feature_DM)
    for i in range(len(ad_degree_FP)):
        feature_FP = calculate_feature(ad_degree_FP[i],cn_degree_FP[i])
        list_degree_FP.append(feature_FP)
    for i in range(len(ad_degree_MF)):
        feature_MF = calculate_feature(ad_degree_MF[i],cn_degree_MF[i])
        list_degree_MF.append(feature_MF)

    for i in range(len(ad_degree_DM)):
        threshold_feature_ad = np.concatenate([ad_degree_DM[i],ad_degree_FP[i],ad_degree_MF[i]],axis=1)
        threshold_feature_cn = np.concatenate([cn_degree_DM[i],cn_degree_FP[i],cn_degree_MF[i]],axis=1)
        threshold_feature = calculate_feature(threshold_feature_ad,threshold_feature_cn)
        degree_threshold_list.append(threshold_feature)
    '''DM+FP+MF'''
    feature_degree = np.vstack((degree_threshold_list))
    '''SNBG'''
    #degree calculate
    ad_degree_DM_snbg = degree_matrix_list(ad_thresh_cut_DM_snbg)
    ad_degree_MF_snbg = degree_matrix_list(ad_thresh_cut_MF_snbg)
    ad_degree_FP_snbg = degree_matrix_list(ad_thresh_cut_FP_snbg)
    cn_degree_DM_snbg = degree_matrix_list(cn_thresh_cut_DM_snbg)
    cn_degree_MF_snbg = degree_matrix_list(cn_thresh_cut_MF_snbg)
    cn_degree_FP_snbg = degree_matrix_list(cn_thresh_cut_FP_snbg)
    #network's degree features
    list_degree_DM_snbg = []
    list_degree_FP_snbg = []
    list_degree_MF_snbg = []
    degree_threshold_list_snbg = []

    for i in range(len(ad_degree_DM_snbg)):
        feature_DM_snbg = calculate_feature(ad_degree_DM_snbg[i],cn_degree_DM_snbg[i])
        list_degree_DM_snbg.append(feature_DM_snbg)
    for i in range(len(ad_degree_FP_snbg)):
        feature_FP_snbg = calculate_feature(ad_degree_FP_snbg[i],cn_degree_FP_snbg[i])
        list_degree_FP_snbg.append(feature_FP_snbg)
    for i in range(len(ad_degree_MF_snbg)):
        feature_MF_snbg = calculate_feature(ad_degree_MF_snbg[i],cn_degree_MF_snbg[i])
        list_degree_MF_snbg.append(feature_MF_snbg)

    for i in range(len(ad_degree_DM_snbg)):
        threshold_feature_ad_snbg = np.concatenate([ad_degree_DM_snbg[i],ad_degree_FP_snbg[i],ad_degree_MF_snbg[i]],axis=1)
        threshold_feature_cn_snbg = np.concatenate([cn_degree_DM_snbg[i],cn_degree_FP_snbg[i],cn_degree_MF_snbg[i]],axis=1)
        threshold_feature_snbg = calculate_feature(threshold_feature_ad_snbg,threshold_feature_cn_snbg)
        degree_threshold_list_snbg.append(threshold_feature_snbg)
    '''DM+FP+MF'''
    feature_degree_snbg = np.vstack((degree_threshold_list_snbg))

    pcc_degree = libsvm(feature_degree,cost) # your cost
    snbg_degree = libsvm(feature_degree_snbg,cost) # your cost
    return pcc_degree,snbg_degree