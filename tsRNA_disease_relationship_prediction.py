import os
# 临时目录避免中文路径导致 joblib/loky 问题
_ascii_temp = r"C:\joblib_temp"
os.makedirs(_ascii_temp, exist_ok=True)
os.environ["JOBLIB_TEMP_FOLDER"] = _ascii_temp
os.environ["TMP"] = _ascii_temp
os.environ["TEMP"] = _ascii_temp

import sys, math, random, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedKFold, RandomizedSearchCV, cross_validate,ParameterSampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# -------- 配置区（按需修改） --------
CSV_PATH = r'd:\minecraft\versions\1.18.2forge-newhorizon\kubejs\server_scripts\tsRNA_Disease.csv'
RANDOM_STATE = 42             # 用于模型训练
SAMPLE_SEED = None            # 抽样种子：None 表示每次抽样不同
RANDOM_SEARCH_ITERS = 24    # 随机搜索迭代次数
TEST_SIZE = 0.2
CV_SPLITS = 5
CV_REPEATS = 2
NEG_RATIO = 3                 # 负样本与正样本比例（修改以控制负样本数量）


# -------- 可复现与随机器设置 --------
if SAMPLE_SEED is not None:
    random.seed(SAMPLE_SEED)
    np.random.seed(SAMPLE_SEED)
rng = random.Random(SAMPLE_SEED) if SAMPLE_SEED is not None else random

# -------- 载入数据 --------
df = None
for enc in ('utf-8', 'utf-8-sig', 'gbk'):
    try:
        df = pd.read_csv(CSV_PATH, engine='python', encoding=enc)
        break
    except Exception:
        df = None
if df is None:
    raise SystemExit(f"无法读取 CSV: {CSV_PATH}")

df.columns = [c.strip() for c in df.columns]

# 规范列名
col_map = {}
for c in df.columns:
    lc = c.lower()
    if ('rna' in lc and 'symbol' in lc) or lc.startswith('rna'):
        col_map[c] = 'RNA'
    if ('disease' in lc and 'name' in lc) or lc.startswith('disease') or 'disease' in lc:
        col_map[c] = 'Disease'
    if 'score' in lc:
        col_map[c] = 'Score'
    if 'species' in lc:
        col_map[c] = 'Species'
df = df.rename(columns=col_map)

if 'RNA' not in df.columns or 'Disease' not in df.columns:
    raise SystemExit("CSV 必须包含 RNA 与 Disease 列")

# Score 处理：若不存在则默认 0.0；若存在则转数值
if 'Score' in df.columns:
    df['Score'] = pd.to_numeric(df['Score'], errors='coerce').fillna(1.0).clip(lower=0.0)
else:
    df['Score'] = 0.0

df['RNA'] = df['RNA'].astype(str).str.strip()
df['Disease'] = df['Disease'].astype(str).str.strip()

print("原始数据信息:")
print(f"  行数: {len(df)}, 唯一 tsRNA: {df['RNA'].nunique()}, 唯一 disease: {df['Disease'].nunique()}")
print("Score 描述：")
print(df['Score'].describe())

# -------- 预建邻居集合，快速相似度计算（基于正样本 df） --------
tsrna_to_diseases = {}
disease_to_tsrnas = {}
for _, row in df.iterrows():
    t = row['RNA']; d = row['Disease']
    tsrna_to_diseases.setdefault(t, set()).add(d)
    disease_to_tsrnas.setdefault(d, set()).add(t)

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
pairs_df = df[['RNA', 'Disease', 'Score']].copy()
N = len(pairs_df)
j_list = []
start = time.time()
for idx, row in pairs_df.iterrows():
    if idx % 1000 == 0 and idx > 0:
        elapsed = time.time() - start
        print(f"  进度 {idx}/{N} ({elapsed:.1f}s)")
    j_list.append(fast_jaccard_feature(row['RNA'], row['Disease']))
pairs_df['jaccard_feat'] = j_list

# -------- 构建正样本特征表（对观测对使用 LOO 聚合） --------
# LOO 统计特征（对每条观测，计算去掉该观测后的组均值）
global_mean = df['Score'].mean()
grp_rna_count = df.groupby('RNA')['Score'].transform('count')
grp_rna_sum = df.groupby('RNA')['Score'].transform('sum')
pairs_df['tsrna_score_mean'] = np.where(grp_rna_count > 1,
                                        (grp_rna_sum - pairs_df['Score']) / (grp_rna_count - 1),
                                        global_mean)
pairs_df['tsrna_score_max'] = df.groupby('RNA')['Score'].transform('max')
pairs_df['tsrna_disease_count'] = df.groupby('RNA')['Disease'].transform('nunique')

grp_dis_count = df.groupby('Disease')['Score'].transform('count')
grp_dis_sum = df.groupby('Disease')['Score'].transform('sum')
pairs_df['disease_score_mean'] = np.where(grp_dis_count > 1,
                                         (grp_dis_sum - pairs_df['Score']) / (grp_dis_count - 1),
                                         global_mean)
pairs_df['disease_score_max'] = df.groupby('Disease')['Score'].transform('max')
pairs_df['disease_tsrna_count'] = df.groupby('Disease')['RNA'].transform('nunique')

# # 填充缺失值
# for col in ['tsrna_disease_count','tsrna_score_mean','tsrna_score_max',
#             'disease_tsrna_count','disease_score_mean','disease_score_max']:
#     if col in pairs_df.columns:
#         pairs_df[col] = pairs_df[col].fillna(global_mean if 'score' in col else 0)

# 衍生特征
pairs_df['count_ratio'] = pairs_df['tsrna_disease_count'] / (pairs_df['disease_tsrna_count'] + 1)
pairs_df['score_mean_diff'] = pairs_df['tsrna_score_mean'] - pairs_df['disease_score_mean']

# -------- 负样本构建（随机采样未观测对） --------
observed_set = set(map(tuple, df[['RNA','Disease']].values.tolist()))
all_tsrna = df['RNA'].unique().tolist()
all_diseases = df['Disease'].unique().tolist()
n_pos = len(pairs_df)
max_possible_neg = max(len(all_tsrna)*len(all_diseases) - len(observed_set), 0)
need_neg = min(int(n_pos * NEG_RATIO), max_possible_neg)

neg_samples = set()
attempts = 0
max_attempts = need_neg * 10 + 1000
while len(neg_samples) < need_neg and attempts < max_attempts:
    t = rng.choice(all_tsrna)
    d = rng.choice(all_diseases)
    if (t, d) in observed_set:
        attempts += 1
        continue
    neg_samples.add((t, d))
    attempts += 1

neg_list = list(neg_samples)
print(f"构建负样本: 需要 {need_neg}，实际采样 {len(neg_list)}（最大可能 {max_possible_neg}）")

if len(neg_list) > 0:
    neg_df = pd.DataFrame(neg_list, columns=['RNA','Disease'])
    neg_df['Score'] = 0.0
    # jaccard for negatives (based on positive graph)
    neg_df['jaccard_feat'] = [fast_jaccard_feature(r, d) for r, d in zip(neg_df['RNA'], neg_df['Disease'])]

    # 计算基于正样本的组统计（非 LOO，因为负样本不在正样本中）
    rna_stats = df.groupby('RNA')['Score'].agg(['mean','max','nunique']).rename(columns={'mean':'rna_mean','max':'rna_max','nunique':'rna_nunique'})
    dis_stats = df.groupby('Disease')['Score'].agg(['mean','max','nunique']).rename(columns={'mean':'dis_mean','max':'dis_max','nunique':'dis_nunique'})

    neg_df = neg_df.merge(rna_stats, left_on='RNA', right_index=True, how='left')
    neg_df = neg_df.merge(dis_stats, left_on='Disease', right_index=True, how='left')

    neg_df['rna_mean'] = neg_df['rna_mean'].fillna(global_mean)
    neg_df['rna_max'] = neg_df['rna_max'].fillna(0.0)
    neg_df['rna_nunique'] = neg_df['rna_nunique'].fillna(0).astype(int)
    neg_df['dis_mean'] = neg_df['dis_mean'].fillna(global_mean)
    neg_df['dis_max'] = neg_df['dis_max'].fillna(0.0)
    neg_df['dis_nunique'] = neg_df['dis_nunique'].fillna(0).astype(int)

    neg_df = neg_df.rename(columns={
        'rna_mean':'tsrna_score_mean','rna_max':'tsrna_score_max','rna_nunique':'tsrna_disease_count',
        'dis_mean':'disease_score_mean','dis_max':'disease_score_max','dis_nunique':'disease_tsrna_count'
    })

    neg_df['count_ratio'] = neg_df['tsrna_disease_count'] / (neg_df['disease_tsrna_count'] + 1)
    neg_df['score_mean_diff'] = neg_df['tsrna_score_mean'] - neg_df['disease_score_mean']

    # 与正样本合并
    featured_df = pd.concat([pairs_df, neg_df[[
        'RNA','Disease','Score','jaccard_feat','tsrna_score_mean','tsrna_score_max','tsrna_disease_count',
        'disease_tsrna_count','disease_score_mean','disease_score_max','count_ratio','score_mean_diff'
    ]]], ignore_index=True, sort=False)
else:
    featured_df = pairs_df.copy()

# 方便后续查找单个 tsRNA/disease 的统计信息（基于非泄露逻辑）
tsrna_features = featured_df[['RNA','tsrna_disease_count','tsrna_score_mean','tsrna_score_max']].drop_duplicates().reset_index(drop=True)
disease_features = featured_df[['Disease','disease_tsrna_count','disease_score_mean','disease_score_max']].drop_duplicates().reset_index(drop=True)

#准备机器学习模型的输入数据
feature_cols = ['jaccard_feat','tsrna_disease_count','tsrna_score_mean','tsrna_score_max',
                'disease_tsrna_count','disease_score_mean','disease_score_max',
                'count_ratio','score_mean_diff']
X = featured_df[feature_cols].values
y = featured_df['Score'].astype(float).values

print("特征维度:", X.shape[1], "样本数:", X.shape[0])

# -------- 划分训练/测试并训练回归模型（保留 CV 与随机搜索） --------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

pipe = Pipeline([('scaler', StandardScaler()), ('reg', RandomForestRegressor(random_state=RANDOM_STATE))])
param_dist = {
    'reg__n_estimators': [100, 300, 500],
    'reg__max_depth': [None, 10, 30],
    'reg__min_samples_split': [2, 5, 10],
    'reg__max_features': [None, 'sqrt', 'log2', 0.5]
}

print("开始随机搜索（n_jobs=1，避免并行问题）...")
param_list = list(ParameterSampler(param_dist, n_iter=RANDOM_SEARCH_ITERS, random_state=RANDOM_STATE))
n_candidates = len(param_list)
best_score = -np.inf
best_params = None

search_start = time.time()
for idx, params in enumerate(param_list, start=1):
    cand_start = time.time()
    pipe_cand = Pipeline([('scaler', StandardScaler()), ('reg', RandomForestRegressor(random_state=RANDOM_STATE))])
    pipe_cand.set_params(**params)
    cv_res = cross_validate(pipe_cand, X_train, y_train, cv=3, scoring='neg_root_mean_squared_error', n_jobs=1, return_train_score=False)
    mean_score = np.mean(cv_res['test_score'])
    cand_time = time.time() - cand_start
    elapsed = time.time() - search_start
    avg_time = elapsed / idx
    eta = avg_time * (n_candidates - idx)
    print(f"[{idx}/{n_candidates}] mean_neg_RMSE={mean_score:.6f}  params={params}  time={cand_time:.1f}s  elapsed={elapsed:.1f}s  ETA={eta:.1f}s")
    if mean_score > best_score:
        best_score = mean_score
        best_params = params

print(f"搜索完成，总耗时 {time.time()-search_start:.1f}s，最佳 mean_neg_RMSE={best_score:.6f}，最佳参数={best_params}")
# 使用最佳参数训练最终管道（在训练集上）
best = Pipeline([('scaler', StandardScaler()), ('reg', RandomForestRegressor(random_state=RANDOM_STATE))])
if best_params is not None:
    best.set_params(**best_params)
best.fit(X_train, y_train)


# 在测试集上评估
y_pred = best.predict(X_test)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\n测试集评估（留出集）:")
print(f"  RMSE: {rmse:.6f}")
print(f"  MAE : {mae:.6f}")
print(f"  R2  : {r2:.6f}")

# 更稳健的交叉验证评估（重复 K 折）
rk = RepeatedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)
scoring = {'RMSE': 'neg_root_mean_squared_error', 'MAE': 'neg_mean_absolute_error', 'R2': 'r2'}
cv_res = cross_validate(best, X, y, cv=rk, scoring=scoring, n_jobs=1, return_train_score=False)
print("\n交叉验证结果（RepeatedKFold）:")
print(f"  RMSE mean: { -np.mean(cv_res['test_RMSE']) :.6f}  std: { np.std([-v for v in cv_res['test_RMSE']]) :.6f}")
print(f"  MAE  mean: { -np.mean(cv_res['test_MAE']) :.6f}  std: { np.std([-v for v in cv_res['test_MAE']]) :.6f}")
print(f"  R2   mean: { np.mean(cv_res['test_R2']) :.6f}  std: { np.std(cv_res['test_R2']) :.6f}")

# 训练在全部观测对上以便任意对预测（若需要）
best.fit(X, y)

# -------- 辅助：构造单对特征并预测的函数 --------
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

# -------- 随机抽样输出（用于快速检验）--------
# -------- 随机抽样输出（每次随机两组 tsRNA-疾病对）--------
tsrna_list = featured_df['RNA'].unique().tolist()
disease_list = featured_df['Disease'].unique().tolist()
if len(tsrna_list) >= 2 and len(disease_list) >= 2:
    t1 = rng.choice(tsrna_list)
    d1 = rng.choice(disease_list)
    t2 = rng.choice(tsrna_list)
    d2 = rng.choice(disease_list)
    p1 = predict_score(t1, d1, best)
    p2 = predict_score(t2, d2, best)
    print("\n随机抽取两组 tsRNA-疾病对：")
    print(f"  预测 {t1} 与 {d1} 的相关性 = {p1:.6f}")
    print(f"  预测 {t2} 与 {d2} 的相关性 = {p2:.6f}")

# -------- 手动预测区：读取 PREDICT_PAIRS 并输出（在此处编辑以单独验证）--------
# 手动预测对（在代码区修改以单独验证）
PREDICT_PAIRS = [
     ('Gly-tRF', 'Hepatocellular Carcinoma'),
]
if PREDICT_PAIRS:
    print("\n手动预测结果：")
    for (rna, disease) in PREDICT_PAIRS:
        score = predict_score(rna, disease, best)
        print(f"  预测 {rna} 与 {disease} 的相关性 = {score:.6f}")
else:
    print("\nPREDICT_PAIRS 列表为空：如需单独预测，在代码区修改 PREDICT_PAIRS 并重新运行。")



from joblib import dump

MODEL_PATH = r'd:\minecraft\versions\1.18.2forge-newhorizon\kubejs\server_scripts\tsRNA_disease_predict.joblib'

# 将可复现特征所需的数据结构全部保存
meta = {
    'tsrna_features': tsrna_features,
    'disease_features': disease_features,
    'global_mean': global_mean,
    'feature_cols': feature_cols,
    # 保持集合序列化友好：将 set 转为 list
    'tsrna_to_diseases': {k: list(v) for k, v in tsrna_to_diseases.items()},
    'disease_to_tsrnas': {k: list(v) for k, v in disease_to_tsrnas.items()},
    #可选
    'featured_df': featured_df,   # 可注释掉以减少文件体积
    'orig_df_columns': df.columns.tolist()
}

dump({'model': best, 'meta': meta}, MODEL_PATH)
print(f"已保存模型与元数据到: {MODEL_PATH}")