from datapackmutriku.data_utils import *
import pandas as pd
import numpy as np

x = np.array([1,2,34,4,1,6,22,8])
v = np.array([1,2,34,4,1,6,22,8])
num_bins = 5

data = {'Animal': [1, 2, 1, 3],
        'Age': [5, 3, 8, 2],
        'Weight': [49.5, 35.2, 60.6, 70.0],
        'Height': [50.5, 40.2, 60.1, 70.0],
        'Year': [2020, 2021, 2022, 2023],
        'ID': [1001, 1002, 1003, 1004],  
        'Score': [85.5, 92.2, 75.1, 80.0]}  
df1 = pd.DataFrame(data)

# Crear dataframe con valores discretos y continuos y una columna de etiquetas binarias
df2 = pd.DataFrame({'A': [1, 2, 3, 4],
                   'B': [5.6, 6.6, 7.4, 8.4],
                   'C': [9, 10, 9, 12],
                   'D': [13.9, 14.3, 1.0, 15.5],
                   'labels': [0, 1, 0, 1]})
atributo = "D"
labels = "labels"
condition = "Variance>1"

def test_all():
    """Executes the tests to verify the installation has ben successful (does not test auxiliary functions)"""
    discretize_ew(x, num_bins)
    discretize_ew_manual(x, num_bins)
    discretize_ef(x, 2)
    discretize_ef_manual(x, 4)
    discretize_dataset(df1, num_bins)
    normalize_data(v)        
    normalize_dataframe(df1)
    standardize(v)
    standardize_dataframe(df1)
    calc_entropy(x)
    calc_variance(x)
    calc_AUC(df2, atributo, labels)
    calc_metrics_df(df2, labels)
    filter_df(df2, condition, labels)
    calc_correlation(x, v)
    calc_MI(x,v,num_bins)
    res1, res2 = metricas_pares(df2)
    plot_MI(res1)
    plot_cor(res2)
    metricas_pares(df2, True)