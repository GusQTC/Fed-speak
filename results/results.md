new model results



# redone with the new data
#                            feature  importance
Inflation Rate    0.405087
Mean_Sentiment_Score_Corrected    0.134630
Treasury Rate    0.129584
Median_Sentiment_Score_Corrected    0.119303
StdDev_Sentiment_Score_Corrected    0.108258
Unemployment Rate    0.103139

Best parameters: {'base_score': 0.75, 'booster': 'gbtree', 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 3, 'n_estimators': 100}

MAE: 0.10118283424200065, MSE: 0.016967208822657696, RMSE: 0.13025823898186


previous 

# importances
#                          feature  importance
Unemployment Rate    3.910008
Mean_Sentiment_Score_Corrected    1.152161
Treasury Rate    0.124349
Inflation Rate    0.000014
GDP Change   -4.186532


new
MAE: 6.6317340698622225, MSE: 77.96373160343194, RMSE: 8.829707333962544 ex se houve aumento de .25, o erro seria de .08 entre .32 e .17
feature                             importance
Median_Sentiment_Score_Corrected    0.146155
Mean_Positive_Corrected             0.138514
Mean_File_Count                     0.132519
Inflation Rate                      0.132478
Mean_Word_Count                     0.096188
Mean_Negative_Corrected             0.092316
Treasury Rate                       0.082635
Mean_Sentiment_Score_Corrected      0.062124
GDP Change                          0.051189
StdDev_Sentiment_Score_Corrected    0.047798
Unemployment Rate                   0.018084