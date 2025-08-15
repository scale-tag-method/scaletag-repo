import pandas as pd
import numpy as np
from numpy.linalg import norm

def calcCossim(row, out_df):
    sum_cossim = 0
    for index, row_ref in out_df.iterrows():
        sum_cossim += np.dot(row, row_ref) / (norm(row) * norm(row_ref))
    return sum_cossim

def calcCossimCV(row, out_df, cnt):
    cossim = []
    for index, row_ref in out_df.iterrows():
        cossim.append(np.dot(row, row_ref) / (norm(row) * norm(row_ref)) )

    if cnt > 7:
        set_cv = np.std(cossim)/np.mean(cossim)
    else:
        set_cv = 1/sum(cossim)
    return set_cv

def calculate_cos_sim_df(out_df):
    out_df_cs = out_df.copy()
    out_df_cs['cossimVal'] = 0
    out_df_cs['cossimVal'] = out_df_cs.apply(lambda row: calcCossim(row, out_df_cs), axis=1)
    return out_df_cs

def build_dfs(first_df, second_df, out_df, act_name_list, cnt):
    out_df_cs = out_df.copy()
    out_df_cs['cossimVal1'] = 0
    out_df_cs['cossimVal2'] = 0

    for index, row_cs in out_df_cs.iterrows():
        cossimVal1 = calcCossimCV(row_cs[act_name_list], first_df[act_name_list], cnt)# , a =1)
        cossimVal2 = calcCossimCV(row_cs[act_name_list], second_df[act_name_list], cnt)#, a =2)
        out_df_cs.loc[index, 'cossimVal1'] = cossimVal1
        out_df_cs.loc[index, 'cossimVal2'] = cossimVal2

    min_index_1 = out_df_cs['cossimVal1'].idxmax()
    first_df = first_df.append(out_df.loc[min_index_1])
    out_df_cs = out_df_cs.drop(min_index_1)
    out_df = out_df.drop(min_index_1)

    if len(out_df.index) > 0:
        min_index_2 = out_df_cs['cossimVal2'].idxmax()
        second_df = second_df.append(out_df.loc[min_index_2])
        out_df = out_df.drop(min_index_2)

    if len(out_df.index) == 0:
        return first_df, second_df, out_df
    else:
        cnt += 1
        return build_dfs(first_df, second_df, out_df, act_name_list, cnt)


if __name__ == '__main__':
    
    act_name_list = ["EMS", "Miamij", "Manual in Line", "MBP", "BK", "Oxygen"]

    out_df = pd.read_csv("act_dist_set1_test.csv")
    out_df_cossim = calculate_cos_sim_df(out_df)
    first_min_index = out_df_cossim['cossimVal'].idxmin()
    out_df_cossim_copy = out_df_cossim.drop(first_min_index)
    second_min_index = out_df_cossim_copy['cossimVal'].idxmin()

    first_df = pd.DataFrame(columns=act_name_list)
    second_df = pd.DataFrame(columns=act_name_list)
    first_df = first_df.append(out_df.loc[first_min_index])
    second_df = second_df.append(out_df.loc[second_min_index])
    out_df = out_df.drop(first_min_index)
    out_df = out_df.drop(second_min_index)
    first_df, second_df, out_df = build_dfs(first_df, second_df, out_df, act_name_list, cnt = 0)

    first_df['Unnamed: 0'] = first_df['Unnamed: 0'].astype(int)
    second_df['Unnamed: 0'] = second_df['Unnamed: 0'].astype(int)
    first_df = first_df.sort_index(axis=0)
    second_df = second_df.sort_index(axis=0)
    print("samples for co1 subset:")
    print(first_df['Unnamed: 0'].to_numpy())
    print("samples for co2 subset:")
    print(second_df['Unnamed: 0'].to_numpy())
    first_df.to_csv('model_co1_test.csv')
    second_df.to_csv('model_co2_test.csv')
    print("finished!")

