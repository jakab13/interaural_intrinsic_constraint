from Analysis.plotting_panels import save_panels_for_experiment

# Combined-cue, 1300 Hz (your example)
png_path = save_panels_for_experiment(
    csv_path="Dataframes/psychometrics_params_selected_model_pred.csv",
    dataset="combined_cue",
    frequency=1300.0,
    out_dir="Plots"
)

# If you want 500 Hz:
# save_panels_for_experiment(dataset="combined_cue", frequency=500.0)

# For single_cue or across_frequencies (no frequency parameter):
# save_panels_for_experiment(dataset="single_cue", frequency=None)
# save_panels_for_experiment(dataset="across_frequencies", frequency=None)
