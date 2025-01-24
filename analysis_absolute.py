# 1. read in all the results files per subject

import os
import psignifit as ps
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels

path = "./"
os.chdir(path)
# Read text File
def read_text_file(file_path):
    with open(file_path, 'r') as f:
        print(f.read())

for file in os.listdir():
    if file.endswith(".txt"):
        file_path = f"{path}/{file}"
        read_text_file(file_path)


#2. generate one large "master" file that contains all the recordings of the experiments

# processing individual text files
def get_data(file_path):
    data = open(file_path).read()
    data = data.split('\n')
    data = [line.split(',') for line in data if line.strip()]
    return data[1:]

df = pd.DataFrame(columns = ['subject','datetime_onset','stim_type','trial_type','trial_index','standard_angle','standard_value','standard_cue','standard_center_frequency','comparison_angle','comparison_value','comparison_cue','comparison_center_frequency','standard_order','comparison_order','solution','inter_stimulus_interval','response','is_correct','score', 'score_abs', 'standard_angle_abs','reaction_time', 'comparison_angle_abs'])

folder_path='./Results/pilot_3_angle'
for root, dirs, files in os.walk(folder_path):
    for f in files:
        if ('.txt' in f):
            file_path = os.path.join(root, f)
            data = get_data(file_path)
            trial = f.split('.txt')[0]#.split('kirke')[1]#not necessary

            iter_df = pd.DataFrame(data, index=[trial] * len(data),
                                   columns=['subject', 'datetime_onset', 'stim_type', 'trial_type', 'trial_index',
                                            'standard_angle',
                                            'standard_value', 'standard_cue', 'standard_center_frequency',
                                            'comparison_angle',
                                            'comparison_value', 'comparison_cue', 'comparison_center_frequency',
                                            'standard_order',
                                            'comparison_order', 'solution', 'inter_stimulus_interval', 'response',
                                            'is_correct',
                                            'score', 'score_abs', 'standard_angle_abs', 'reaction_time', 'comparison_angle_abs'
                                            ])

            df = pd.concat([df, iter_df])
        else:
            pass
#df['standard_angle_abs'] = df['standard_angle_abs'].astype(str).astype(int)

combined_df = df.rename_axis('trial').reset_index()
combined_df = combined_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
combined_df['comparison_angle_abs'] = pd.to_numeric(combined_df['comparison_angle_abs'], errors='coerce')
combined_df['score_abs'] = pd.to_numeric(combined_df['score_abs'], errors='coerce')
combined_df.sort_values(by='standard_center_frequency')
print(combined_df)
combined_df.to_csv('./merged_1.txt', sep='\t', index=False)


# 3. calculate the psychometric function for each participant, for each "trial_type", for each "standard_angle"
#       use the psignifit library for the calculations


def get_psychometric_means(df_group):
    data = df_group.groupby("comparison_angle_abs", as_index=False).agg(
        {"score_abs": "sum", df_group.columns[0]: "count"}).rename(
        columns={df_group.columns[0]: 'n_total'})
    res = ps.psignifit(
        data.values,
        sigmoid="logistic",
        experiment_type="equal asymptote"
    )
    return res.parameter_estimate   #parameters_estimate_mean

df_group = combined_df.groupby(["subject", "standard_angle_abs", "standard_center_frequency", "trial_type"])
#df_model = df_group.apply(lambda g: get_psychometric_means(g)["threshold"]).reset_index(name='threshold')
#df_model = df_group.apply(lambda g: get_psychometric_means(g)["width"]).reset_index(name='width')

df_model = df_group.apply(
    lambda g: pd.Series({
        'threshold': get_psychometric_means(g)["threshold"],
        'width': get_psychometric_means(g)["width"]})
    ).reset_index()


# 4. store the results (mean, slope) of the psychometric fit in the same data sheet
df_model.to_csv('./psychometric_results.csv', index=False)

# 5. plot psychometric functions in a Facetgrid using the seaborn library


#sns.FacetGrid(combined_df, col="standard_angle", row="standard_center_frequency", hue="trial_type", )#.map(plt.plot, "standard_angle", "threshold")
palette= {'ILD-->ILD':'C1','ITD-->ITD':'C0','ILD-->ITD':'C2','ITD-->ILD':'C3'}
pf = sns.FacetGrid(combined_df, row="standard_angle_abs", col="standard_center_frequency", hue="trial_type", margin_titles=True,
                   col_order=['600','1200'], row_order=['8','16'], hue_order=['ILD-->ILD','ITD-->ITD','ILD-->ITD','ITD-->ILD'],
                   palette=palette)
pf.map(sns.lineplot, "comparison_angle_abs", "score_abs", marker="o")
#pf.map(sns.regplot, "comparison_angle_abs", "score_abs", logistic=True, scatter=False, marker="o") #logistic- von 2 antworten funktion bilden, nur zwei y werte
pf.set_xlabels(label='Comparison angle')
pf.set_ylabels(label='Score')
pf.set(xticks=[-45, -35, -25, -15, 0, 15,25,35,45])
pf.add_legend()
axes=pf.axes.flatten()
#axes[0].set_title("400 Hz")
#axes[0].set_title("500 Hz")
axes[0].set_title("600 Hz")
#axes[3].set_title("700 Hz")
axes[1].set_title("1200 Hz")
pf.savefig('./figures/pilot_3_angle_points')
plt.show()


#5. plot width against frequencies for all subjects in one plot, each width is one point
palette= {'ILD-->ILD':'C1','ITD-->ITD':'C0','ILD-->ITD':'C2','ITD-->ILD':'C3'}

sns.lineplot(    #lineplot
    data=df_model,
    x='standard_angle_abs',
    y='width',
    style='standard_center_frequency',
    hue='trial_type',
    err_style='bars',
    errorbar='ci',
    marker='o',
    palette=palette
)

plt.savefig('./figures/width_pilot_3')
plt.show()
