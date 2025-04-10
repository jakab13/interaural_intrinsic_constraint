# 1. read in all the results files per subject

import os
import psignifit as ps
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels

path = "/home/kirke/Schreibtisch/ma/python/cue_integration/Results/pilot/pilot_4"
path = "/Users/jakabpilaszanovich/Documents/GitHub/interaural_intrinsic_constraint/Results/jakab_mixing_gain"
# Change the directory
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
# merging dataframe
df = pd.DataFrame(columns = ['subject','datetime_onset','stim_type','trial_type','trial_index','standard_angle','standard_value','standard_cue','standard_center_frequency','comparison_angle','comparison_value','comparison_cue','comparison_center_frequency','standard_order','comparison_order','solution','inter_stimulus_interval','response','is_correct','score','standard_angle_abs','reaction_time'])

folder_path='/home/kirke/Schreibtisch/ma/python/cue_integration/Results/pilot/pilot_4'
for root, dirs, files in os.walk(folder_path):
    for f in files:
        if ('.txt' in f):
            file_path = os.path.join(root, f)
            data = get_data(file_path)
            trial = f.split('.txt')[0]

            iter_df = pd.DataFrame(data, index=[trial] * len(data),
                                   columns=['subject', 'datetime_onset', 'stim_type', 'trial_type', 'trial_index',
                                            'standard_angle',
                                            'standard_value', 'standard_cue', 'standard_center_frequency',
                                            'comparison_angle',
                                            'comparison_value', 'comparison_cue', 'comparison_center_frequency',
                                            'standard_order',
                                            'comparison_order', 'solution', 'inter_stimulus_interval', 'response',
                                            'is_correct',
                                            'score', 'standard_angle_abs', 'reaction_time'])

            df = pd.concat([df, iter_df])
        else:
            pass

combined_df = df.rename_axis('trial').reset_index()
combined_df = combined_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
combined_df['comparison_angle'] = pd.to_numeric(combined_df['comparison_angle'], errors='coerce')
combined_df['score'] = pd.to_numeric(combined_df['score'], errors='coerce')
combined_df.sort_values(by='standard_center_frequency')
print(combined_df)
combined_df.to_csv('/home/kirke/Schreibtisch/ma/python/cue_integration/merged.txt', sep='\t', index=False)


# 3. calculate the psychometric function for each participant, for each "trial_type", for each "standard_angle"
#       use the psignifit library for the calculations

def get_psychometric_means(df_group):
    data = df_group.groupby("comparison_angle", as_index=False).agg(
        {"score": "sum", df_group.columns[0]: "count"}).rename(
        columns={df_group.columns[0]: 'n_total'})
    res = ps.psignifit(
        data.values,
        sigmoid="logistic",
        experiment_type="equal asymptote"
    )
    return res.parameter_estimate   #parameters_estimate_mean


df_group = combined_df.groupby(["subject", "standard_angle", "standard_center_frequency", "trial_type"])
#df_model = df_group.apply(lambda g: get_psychometric_means(g)["threshold"]).reset_index(name='threshold')
#df_model = df_group.apply(lambda g: get_psychometric_means(g)["width"]).reset_index(name='width')

df_model = df_group.apply(
    lambda g: pd.Series({
        'threshold': get_psychometric_means(g)["threshold"],
        'width': get_psychometric_means(g)["width"]})
    ).reset_index()


# 4. store the results (mean, slope) of the psychometric fit in the same data sheet
df_model.to_csv('/home/kirke/Schreibtisch/ma/python/cue_integration/psychometric_results.csv', index=False)

# 5. plot psychometric functions in a Facetgrid using the seaborn library


#sns.FacetGrid(combined_df, col="standard_angle", row="standard_center_frequency", hue="trial_type", )#.map(plt.plot, "standard_angle", "threshold")
palette= {'ILD-->ILD':'C1','ITD-->ITD':'C0'}
pf = sns.FacetGrid(combined_df, row="standard_angle", col="standard_center_frequency", hue="trial_type", margin_titles=True, col_order=['400','500','600','700','800'], row_order=['0'], hue_order=['ILD-->ILD','ITD-->ITD'], palette=palette) # palette='husl',
pf.map(sns.lineplot, "comparison_angle", "score", marker="o")
#pf.map(sns.regplot, "comparison_angle", "score", logistic=True, scatter=False, marker="o") #logistic- von 2 antworten funktion bilden, nur zwei y werte
pf.set_xlabels(label='Comparison angle')
pf.set_ylabels(label='Score')
pf.add_legend()
axes=pf.axes.flatten()
axes[0].set_title("400 Hz")
axes[1].set_title("500 Hz")
axes[2].set_title("600 Hz")
axes[3].set_title("700 Hz")
axes[4].set_title("800 Hz")

pf.savefig('/home/kirke/Schreibtisch/ma/python/figures/pilot_4_points')
plt.show()


#6. plot width against frequencies for all subjects in one plot, each width is one point
palette= {'ILD-->ILD':'C1','ITD-->ITD':'C0','ILD-->ITD':'C2','ITD-->ILD':'C3'}

sns.lineplot(    #lineplot
    data=df_model,
    x='standard_center_frequency',
    y='width',
    style='subject',
    hue='trial_type',
    err_style='bars',
    errorbar='ci',
    marker='o',
    palette=palette
)
plt.savefig('/home/kirke/Schreibtisch/ma/python/figures/width')
plt.show()