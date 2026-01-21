from Analysis.model_prep import prepare_selected_with_pse_and_predictions
df = prepare_selected_with_pse_and_predictions(
    "Dataframes/psychometrics_params_selected.csv",
    "Dataframes/psychometrics_params_selected_model_pred.csv"
)

print(df[[
    "dataset","subject_key","trial_type","standard_center_frequency","standard_angle_abs",
    "pse","pse_delta",
    "pse_pred_uncertainty","pse_delta_uncertainty",
    "pse_pred_scaling","pse_delta_scaling"
]].head())
