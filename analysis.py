# read in all the results files per subject
# generate one large "master" file that contains all the recordings of the experiments
# calculate the psychometric function for each participant, for each "trial_type", for each "standard_angle"
#       use the psignifit library for the calculations
# store the results (mean, slope) of the psychometric fit in the same data sheet
# plot psychometric functions in a Facetgrid using the seaborn library


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