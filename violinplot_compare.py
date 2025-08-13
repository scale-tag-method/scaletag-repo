import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="white")
plt.rcParams['text.usetex'] = True

def compare_cotrain_base_all(file_list, activity_names):
    subcat = []
    categ = []
    for i in range(len(activity_names)):
        categ.append(activity_names[i])
        categ.append(activity_names[i])
        subcat.append("co1")
        subcat.append("co2")

    df_case1_list = []
    df_case2_list = []
    model_names = ["Baseline", "Baseline", r"\textsc{scale-tag}", r"\textsc{scale-tag}", "Supervised", "Supervised"]
    for i in range(0,len(file_list),2):
        f1 = open(file_list[i], 'r')
        all_lines1 = f1.readlines()
        f1.close()
        f2 = open(file_list[i+1], 'r')
        all_lines2 = f2.readlines()
        f2.close()

        set_number = len(all_lines1)

        values = []
        for idx in range(set_number):
            my_line1 = all_lines1[idx]
            my_line2 = all_lines2[idx]
            my_line1_lst = my_line1.split()[:6]
            my_line2_lst = my_line2.split()[:6]
            my_line = []
            for j in range(len(my_line1_lst)):
                my_line.append(my_line1_lst[j])
                my_line.append(my_line2_lst[j])
            values.append(my_line)

        values_arr = np.array(values)
        transposed_np = values_arr.T
        transposed_values = transposed_np.tolist()
        score_dict = {"act_name": categ, "cotrain": subcat, "values": transposed_values}
        score_df = pd.DataFrame(score_dict)
        #print(score_df)
        score_df_exploded = score_df.explode('values')
        #print(score_df_exploded)
        df_case1 = score_df_exploded[score_df_exploded["act_name"].isin(["EMS collar \n placement", "Miami-j collar \n placement", "Oxygen \n administration"])]
        df_case1 = df_case1.reset_index(drop=True)
        df_case1["values"] = df_case1["values"].astype(float)
        df_case1['model'] = model_names[i]
        df_case1_list.append(df_case1)

        df_case2 = score_df_exploded[score_df_exploded["act_name"].isin(['Manual in-line \n stabilization', 'Blood pressure \n measurement', 'Back \n examination'])]
        df_case2 = df_case2.reset_index(drop=True)
        df_case2["values"] = df_case2["values"].astype(float)
        df_case2['model'] = model_names[i]
        df_case2_list.append(df_case2)

    combined_df_case1 = pd.concat([df_case1_list[0], df_case1_list[1], df_case1_list[2]])
    combined_df_case2 = pd.concat([df_case2_list[0], df_case2_list[1], df_case2_list[2]])

    fig, ax1 = plt.subplots(figsize=(12, 12))
    plt.rcParams.update({'font.size': 26})
    ax1.xaxis.label.set_fontsize(26)
    ax1.yaxis.label.set_fontsize(26)


    combined_df_case1["model name"] = combined_df_case1['model'].astype(str) + ' - ' + combined_df_case1['cotrain'].astype(str)
    ax = sns.violinplot(y="act_name", x="values", data=combined_df_case1, hue = "model name", orient="h", split=True, legend='full', gap=.2, width=1, inner="quartile", saturation =0.3, palette=['gainsboro', 'darkgray', 'skyblue', 'darkturquoise', 'royalblue', 'mediumblue', 'gainsboro', 'darkgray', 'skyblue', 'darkturquoise', 'royalblue', 'mediumblue', 'gainsboro', 'darkgray', 'skyblue', 'darkturquoise', 'royalblue', 'mediumblue'])
    combined_df_case1_B = combined_df_case1[combined_df_case1["model"] == "Baseline"]
    combined_df_case1_A = combined_df_case1[combined_df_case1["model"] == r"\textsc{scale-tag}"]
    combined_df_case1_S = combined_df_case1[combined_df_case1["model"] == "Supervised"]

    sns.boxplot(y=combined_df_case1_B["act_name"], x=combined_df_case1_B["values"], orient="h", color="peachpuff", linecolor="black", width=0.05, positions=[-0.333,0.667,1.667])#data=combined_df_case1_a1_B["values"], orient="h")
    sns.boxplot(y=combined_df_case1_A["act_name"], x=combined_df_case1_A["values"], orient="h", color="lightsalmon", linecolor="black", width=0.05, positions=[0,1,2])
    sns.boxplot(y=combined_df_case1_S["act_name"], x=combined_df_case1_S["values"], orient="h", color="tomato", linecolor="black", width=0.05, positions=[0.333,1.333,2.333])

    plt.xticks(fontsize=24)
    plt.yticks(fontsize=22)
    plt.xlabel(r'\textsc{ap} score distribution', ha='right', fontsize=24)
    plt.ylabel(None)#, labelpad=10)
    plt.xlim(0, 1.1)
    plt.ylim(-0.5, 2.5)
    ax1.axhline(y=1.5, color='grey', linestyle='-', linewidth=0.8)
    ax1.axhline(y=0.5, color='grey', linestyle='-', linewidth=0.8)
    ax1.grid(True, axis="x", linestyle="--")
    plt.legend(loc='lower left', fontsize=22)
    plt.tight_layout()
    plt.savefig('actlearn-compare-baseline-cotrain-supervised-violin4_v5.png')
    plt.show()

    plt.figure()
    fig, ax2 = plt.subplots(figsize=(12, 12))
    plt.rcParams.update({'font.size': 26})
    ax2.xaxis.label.set_fontsize(26)
    ax2.yaxis.label.set_fontsize(26)

    combined_df_case2["model name"] = combined_df_case2['model'].astype(str) + ' - ' + combined_df_case2['cotrain'].astype(str)
    ax = sns.violinplot(y="act_name", x="values", data=combined_df_case2, hue = "model name", orient="h", split=True, legend='full', gap=.2, width=1, inner="quartile", saturation =0.3, palette=['gainsboro', 'darkgray', 'skyblue', 'darkturquoise', 'royalblue', 'mediumblue', 'gainsboro', 'darkgray', 'skyblue', 'darkturquoise', 'royalblue', 'mediumblue', 'gainsboro', 'darkgray', 'skyblue', 'darkturquoise', 'royalblue', 'mediumblue'])
    combined_df_case2_B = combined_df_case2[combined_df_case2["model"] == "Baseline"]
    combined_df_case2_A = combined_df_case2[combined_df_case2["model"] == r"\textsc{scale-tag}"]
    combined_df_case2_S = combined_df_case2[combined_df_case2["model"] == "Supervised"]

    sns.boxplot(y=combined_df_case2_B["act_name"], x=combined_df_case2_B["values"], orient="h", color="peachpuff", linecolor="black", width=0.05, positions=[-0.333,0.667,1.667])#data=combined_df_case1_a1_B["values"], orient="h")
    sns.boxplot(y=combined_df_case2_A["act_name"], x=combined_df_case2_A["values"], orient="h", color="lightsalmon", linecolor="black", width=0.05, positions=[0,1,2])
    sns.boxplot(y=combined_df_case2_S["act_name"], x=combined_df_case2_S["values"], orient="h", color="tomato", linecolor="black", width=0.05, positions=[0.333,1.333,2.333])

    plt.xticks(fontsize=24)
    plt.yticks(fontsize=22)
    plt.xlabel(r'\textsc{ap} score distribution', ha='right', fontsize=24)
    plt.ylabel(None)#, labelpad=10)
    plt.xlim(0, 1.1)
    plt.ylim(-0.5, 2.5)
    ax2.axhline(y=1.5, color='grey', linestyle='-', linewidth=0.8)
    ax2.axhline(y=0.5, color='grey', linestyle='-', linewidth=0.8)
    ax2.grid(True, axis="x", linestyle="--")
    plt.legend(loc='upper left', fontsize=22)
    plt.tight_layout()
    plt.savefig('actlearn-compare-baseline-cotrain-supervised-violin5_v5.png')
    plt.show()


if __name__ == '__main__':
    activity_names = ['EMS collar \n placement', 'Miami-j collar \n placement', 'Manual in-line \n stabilization',
                      'Blood pressure \n measurement', 'Back \n examination', 'Oxygen \n administration']
    file_list = [r'co1_baseline.txt',
                 r'co2_baseline.txt',
                 r'co1_scaletag.txt',
                 r'co2_scaletag.txt',
                 r'co1_supervised.txt',
                 r'co2_supervised.txt']
    compare_cotrain_base_all(file_list, activity_names)
