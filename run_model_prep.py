from Analysis.model_prep import prepare_selected_with_pse_and_predictions

df_pred = prepare_selected_with_pse_and_predictions(
    "Dataframes/psychometrics_params_selected.csv",
    "Dataframes/psychometrics_params_selected_model_pred.csv"
)

print(df_pred[[
    "dataset","subject_key","trial_type","standard_center_frequency","standard_angle_abs",
    "pse","pse_pred_uncertainty","pse_pred_scaling","pse_pred_error_uncertainty","pse_pred_error_scaling",
    "jnd","jnd_pred_uncertainty","jnd_pred_scaling","jnd_pred_error_uncertainty","jnd_pred_error_scaling"
]].head())
