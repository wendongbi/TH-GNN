import numpy as np
import json


# prepare dataset
with open('../data_financial/processed_company_data/business_risk_2020_frist3seasons_multi-hot.json', 'r') as fin:
    business_label_dict = json.load(fin)
with open('../data_financial/processed_company_data/finance_risk_2020_frist3seasons_multi-hot.json', 'r') as fin:
    finance_label_dict = json.load(fin)
with open('../data_financial/processed_company_data/company_expert_index_merge_2019final.json', 'r') as fin:
# with open('processed_company_data/company_expert_index_merge.json', 'r') as fin: 
    company_data_bucket = json.load(fin)
############################ remove possible risk companys current in wind_label_table ###############################
with open('../data_financial/possible_non_risk_companys.json', 'r') as fin:
    possbile_non_risk_companys = json.load(fin)
print('../data_financial/possible_non_risk_companys num:', len(possbile_non_risk_companys))


with open('../data_financial/graph_files/nodes/listed_company_id2nameANDidx.json', 'r') as fin:
    listed_company_id2nameANDidx = json.load(fin)
mapper_name2node_id = {}
for _id in listed_company_id2nameANDidx:
    company_name, node_id = listed_company_id2nameANDidx[_id]
    mapper_name2node_id[company_name] = node_id
print(len(mapper_name2node_id))

node_num = len(mapper_name2node_id)
node_dim = len(company_data_bucket[list(company_data_bucket.keys())[0]]) - 1
data_mat = np.zeros((node_num, node_dim))
label_mat = np.ones(node_num).astype(int) * -1 # complete missing lbl as -1
# label_mat = np.zeros(node_num).astype(int) # complete missing lbl as non-risks
print(len(company_data_bucket))
for company_id in company_data_bucket:
    company_name = company_data_bucket[company_id][0]
    if company_name in mapper_name2node_id:
        node_id = mapper_name2node_id[company_name]
        node_feat = company_data_bucket[company_id][1:]
        assert len(node_feat) == data_mat.shape[1]
        data_mat[node_id] = node_feat
        if company_id in business_label_dict or company_id in finance_label_dict:
            label_mat[node_id] = 1
        # elif company_id in possbile_non_risk_companys:
            # label_mat[node_id] = 0
        # elif (label_mat == 0).sum() < 1698:
        else:
            label_mat[node_id] = 0
print('feature-missing node num:', (data_mat.sum(1) == 0).sum())
print('label-missing node num:', (label_mat == -1).sum())
print('pos_num:{}, neg_num:{}'.format((label_mat == 1).sum(), (label_mat == 0).sum()))
print(data_mat.shape, label_mat.shape)

# process nan case
mask_nan = np.isnan(data_mat)
print('original nan rate:', np.sum(mask_nan) / (len(mask_nan) * len(mask_nan[0]))) # nan rate
thres_nan = 0.6
dim_select = []

for idx in range(data_mat.shape[1]):
    nan_rate = np.isnan(data_mat[:, idx]).sum() / (len(data_mat))
    if nan_rate < thres_nan:
        dim_select.append(idx)
    print('dim:{} | nan_rate:{:.2f}%'.format(idx, nan_rate * 100))
print('dim select:', dim_select, len(dim_select))


data_mat = data_mat[:, dim_select]
mask_nan = np.isnan(data_mat)
print('[del_nan_col] nan rate:', np.sum(mask_nan) / (len(mask_nan) * len(mask_nan[0]))) # nan rate

mask_nan = np.isnan(data_mat)
data_mat[mask_nan] = 0

mask_nan = np.isnan(data_mat)
print('processed nan rate:', np.sum(mask_nan) / (len(mask_nan) * len(mask_nan[0]))) # nan rate
mask_inf = np.isinf(data_mat)
print('inf num:', mask_inf.sum())
data_mat[mask_inf] = 0

# # row normalization
def print_dim_statistic(x, thres_range=50):
    dim_select = []
    for idx in range(x.shape[1]):
        x_dim = x[:, idx]
        range_dim = np.abs(x_dim).max() - np.abs(x_dim).mean()
        print('dim:{} | min:{:.3f} | max:{:.3f} | mean:{:.3f} | range:{:.3f}'.format(idx, x_dim.min(), x_dim.max(), x_dim.mean(), range_dim))
        if range_dim < thres_range:
            print(f'dim-{idx} < {thres_range}, so continue.')
            continue
        dim_select.append(idx)
    return dim_select
dim_need_to_norm = print_dim_statistic(data_mat)
from sklearn.preprocessing import Normalizer, KBinsDiscretizer

# bins strategy 2
bins_encoder = KBinsDiscretizer(n_bins=50, encode='ordinal', strategy='quantile')  # uniform, quantile, kmeans
mask_sign = data_mat < 0
data_mat[mask_sign] *= -1
# data_mat[dim_need_to_norm] = bins_encoder.fit_transform(data_mat[dim_need_to_norm])
data_mat = bins_encoder.fit_transform(data_mat)
data_mat[mask_sign] *= -1


print('data_mat.shape:', data_mat.shape)
np.save('financial_statement.npy', data_mat)
np.save('risk_label.npy', label_mat)