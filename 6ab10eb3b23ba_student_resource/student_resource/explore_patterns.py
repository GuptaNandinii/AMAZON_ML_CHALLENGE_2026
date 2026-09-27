import pandas as pd
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

base = r'd:\AMAZON ML CHALLENGE\6ab10eb3b23ba_student_resource\student_resource\dataset\train'

s1 = pd.read_csv(os.path.join(base, 'train_source1.tsv'), sep='\t', encoding='utf-8', nrows=200000)
s2 = pd.read_csv(os.path.join(base, 'train_source2.tsv'), sep='\t', encoding='utf-8', nrows=200000)
s3 = pd.read_csv(os.path.join(base, 'train_source3.tsv'), sep='\t', encoding='utf-8', nrows=200000)
gt = pd.read_csv(os.path.join(base, 'train_ground_truth.tsv'), sep='\t', encoding='utf-8', nrows=200000)

print('S1 nulls:', s1.isnull().sum().to_dict())
print('S2 nulls:', s2.isnull().sum().to_dict())
print('S3 nulls:', s3.isnull().sum().to_dict())
print()

# Build lookup dicts
s2_dict = s2.set_index('entity_id').to_dict('index')
s3_dict = s3.set_index('entity_id').to_dict('index')
s1_dict = s1.set_index('entity_id').to_dict('index')

gt_notnull = gt[gt['matched_entity_ids'].notna()].head(8)
for _, row in gt_notnull.iterrows():
    s1_id = row['source1_entity_id']
    matched_ids = str(row['matched_entity_ids']).split(',')
    if s1_id in s1_dict:
        r1 = s1_dict[s1_id]
        print(f"S1 [{s1_id}]: {r1['business_name']} | {r1['business_address']} | {r1['country']}")
        for m_id in matched_ids[:3]:
            m_id = m_id.strip()
            d = s2_dict if m_id.startswith('S2-') else s3_dict
            if m_id in d:
                mr = d[m_id]
                print(f"  -> [{m_id}]: {mr['business_name']} | {mr['business_address']} | {mr['country']}")
        print()
