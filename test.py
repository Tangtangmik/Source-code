from joblib import load
import pandas as pd
import numpy as np

MODEL_PATH = r'd:\minecraft\versions\1.18.2forge-newhorizon\kubejs\server_scripts\tsRNA_disease_predict.joblib'

loaded = load(MODEL_PATH)
model = loaded['model']
meta = loaded['meta']

# 恢复邻接表为 set 结构（fast_jaccard 需要）
tsrna_to_diseases = {k: set(v) for k, v in (meta.get('tsrna_to_diseases') or {}).items()}
disease_to_tsrnas = {k: set(v) for k, v in (meta.get('disease_to_tsrnas') or {}).items()}

# 恢复聚合特征表（可能已以 DataFrame 保存）
tsrna_features = meta.get('tsrna_features')
if not isinstance(tsrna_features, pd.DataFrame):
    tsrna_features = pd.DataFrame(tsrna_features)

disease_features = meta.get('disease_features')
if not isinstance(disease_features, pd.DataFrame):
    disease_features = pd.DataFrame(disease_features)

global_mean = meta.get('global_mean', 0.0)

def fast_jaccard_feature(rna, disease):
    A = tsrna_to_diseases.get(rna, set())
    B = disease_to_tsrnas.get(disease, set())
    if not A or not B:
        return 0.0
    diseases_of_B = set()
    for t2 in B:
        diseases_of_B.update(tsrna_to_diseases.get(t2, set()))
    inter = len(A & diseases_of_B)
    union = len(A | diseases_of_B) if len(A | diseases_of_B) > 0 else 1
    return inter / union

def construct_feature_vector(rna, disease):
    j = fast_jaccard_feature(rna, disease)
    ts_row = tsrna_features[tsrna_features['RNA'] == rna]
    dis_row = disease_features[disease_features['Disease'] == disease]
    ts_count = int(ts_row['tsrna_disease_count'].values[0]) if not ts_row.empty else 0
    ts_mean = float(ts_row['tsrna_score_mean'].values[0]) if not ts_row.empty else global_mean
    ts_max = float(ts_row['tsrna_score_max'].values[0]) if not ts_row.empty else 0.0
    dis_count = int(dis_row['disease_tsrna_count'].values[0]) if not dis_row.empty else 0
    dis_mean = float(dis_row['disease_score_mean'].values[0]) if not dis_row.empty else global_mean
    dis_max = float(dis_row['disease_score_max'].values[0]) if not dis_row.empty else 0.0
    count_ratio = ts_count / (dis_count + 1)
    score_mean_diff = ts_mean - dis_mean
    return np.array([j, ts_count, ts_mean, ts_max, dis_count, dis_mean, dis_max, count_ratio, score_mean_diff]).reshape(1, -1)

def predict_score(rna, disease, model):
    feat = construct_feature_vector(rna, disease)
    est = model.predict(feat)[0]
    return float(np.clip(est, 0.0, 1.0))

# 示例：在此处填入要预测的 пары 并运行脚本
PREDICT_PAIRS = [
    ('Gly-tRF', 'Hepatocellular Carcinoma'),
    # ('另一个 tsRNA', '另一个疾病'),
]

if __name__ == '__main__':
    for rna, dis in PREDICT_PAIRS:
        score = predict_score(rna, dis, model)
        print(f"预测 {rna} - {dis} 的相关性 = {score:.6f}")