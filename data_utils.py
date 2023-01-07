import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
import plotly
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

def discretize_ew(x, num_bins):
    """
    This function discretizes the values in a list 'x' into a specified number of bins.
    
    Parameters:
        x (list): The list of values to be discretized.
        num_bins (int): The number of bins to use for discretization.
    
    Returns:
        tuple: A tuple containing two elements:
            cuts (list): A list of the cut points used for discretization.
            names (list): A list of strings representing the factorized values.
    """
    #Calcular puntos de corte
    width = (max(x) - min(x)) / num_bins
    cuts = [min(x) + (width * i) for i in range(1, num_bins)]
    eps = 1E-5
    cut_points = np.concatenate([[min(x)-eps], cuts, [max(x)+eps]])
    
    #Calcular factores
    factors = np.digitize(x, cut_points)
    names = ["I" + str(factor) for factor in factors]
    return (cuts, names)
    
def create_facs(values, cut_points):
    """
    This function creates a list of factors based on a list of values and a list of cut points.
    
    For each value in 'values', the function determines which of the cut points it is less than or equal to.
    If a value is less than or equal to a cut point, the corresponding factor is the index of that cut point in the list.
    If a value is greater than all of the cut points, the corresponding factor is the length of the list of cut points.
    
    Parameters:
        values (list): The list of values for which factors will be created.
        cut_points (list): The list of cut points to use for determining the factors.
    
    Returns:
        list: A list of integers representing the factors corresponding to the elements in 'values'.
    """
    facs = []
    
    for value in values:
        for i, cut in enumerate(cut_points):
            
            if value <= cut:
                facs.append(i)
                break
        else:
            facs.append(len(cut_points))
    return facs

def discretize_ew_manual(x, num_bins):
    """
    This function discretizes the values in a list 'x' into a specified number of bins but using the 'create_facs' to factorize the values instead of 'np.digitize'.
    
    Parameters:
        x (list): The list of values to be discretized.
        num_bins (int): The number of bins to use for discretization.
    
    Returns:
        tuple: A tuple containing two elements:
            cuts (list): A list of the cut points used for discretization.
            names (list): A list of strings representing the factorized values.
    """
    #Calcular puntos de corte
    width = (max(x) - min(x)) / num_bins
    cuts = [min(x) + (width * i) for i in range(1, num_bins)]
    eps = 1E-5
    cut_points = np.concatenate([[min(x)-eps], cuts, [max(x)+eps]])
    
    #Calcular factores
    factors = create_facs(x, cut_points)
    names = ["I" + str(factor) for factor in factors]
    return (cuts, names)

def discretize_ef(x, num_bins):
    """
    This function discretizes the values in a list 'x' into a specified number of bins using equal frequency.
    
    The function sorts the values in 'x' and then divides them into 'num_bins' bins such that each bin contains the same number of values.
    
    Parameters:
        x (list): The list of values to be discretized.
        num_bins (int): The number of bins to use for discretization.
    
    Returns:
        tuple: A tuple containing two elements:
            cuts (list): A list of the cut points used for discretization.
            names (list): A list of strings representing the factorized values.
    """
    #Calcular puntos de corte
    x_sorted = sorted(x)
    width = round(len(x) / num_bins)
    cuts = [x_sorted[width * i] for i in range(1, num_bins)]
    eps = 1E-5
    cut_points = np.concatenate([[min(x)-eps], cuts, [max(x)+eps]])
    
    #Calcular en qué fracción queda cada elemento
    factors = np.digitize(x, cut_points)
    names = ["I" + str(factor) for factor in factors]
    return (cuts, names)
    
def create_facs_v2(values, cut_points):
    """
    This function creates a list of factors based on a list of values and a list of cut points.
    
    For each value in 'values', the function determines which of the cut points it is less than.
    If a value is less than a cut point, the corresponding factor is the index of that cut point in the list.
    If a value is greater than all of the cut points or equal to the last point, the corresponding factor is the length of the list of cut points.
    
    Parameters:
        values (list): The list of values for which factors will be created.
        cut_points (list): The list of cut points to use for determining the factors.
    
    Returns:
        list: A list of integers representing the factors corresponding to the elements in 'values'.
    """    
    facs = []
    
    for value in values:
        for i, cut in enumerate(cut_points):
            
            if value < cut:
                facs.append(i)
                break
        else:
            facs.append(len(cut_points))
    return facs

def discretize_ef_manual(x, num_bins):
    """
    This function discretizes the values in a list 'x' into a specified number of bins using equal frequency, but using the 'create_facs' to factorize the values instead of 'np.digitize'.
    
    The function sorts the values in 'x' and then divides them into 'num_bins' bins such that each bin contains the same number of values.
    
    Parameters:
        x (list): The list of values to be discretized.
        num_bins (int): The number of bins to use for discretization.
    
    Returns:
    tuple: A tuple containing two elements:
        cuts (list): A list of the cut points used for discretization.
        names (list): A list of strings representing the factorized values.
    """
    #Calcular puntos de corte
    x_sorted = sorted(x)
    width = round(len(x) / num_bins)
    cuts = [x_sorted[width * i] for i in range(1, num_bins)]
    eps = 1E-5
    cut_points = np.concatenate([[min(x)-eps], cuts, [max(x)+eps]])

    #Calcular en qué fracción queda cada elemento
    factors = create_facs_v2(x, cut_points)
    names = ["I" + str(factor) for factor in factors]
    return (cuts, names)

def discretize_dataset(df, num_bins, fun=discretize_ew):
    """
    This function discretizes a Pandas DataFrame using a specified function.

    The function applies the specified discretization function 'fun' to each column of the DataFrame,
    and returns a new DataFrame with the discretized values.

    Parameters:
        df (pandas.DataFrame): The DataFrame to be discretized.
        num_bins (int): The number of bins to use for discretization.
        fun (function, optional): The function to use for discretization. Defaults to the 'discretize_ew' (equal width) function.

    Returns:
        pandas.DataFrame: A DataFrame with the same structure as 'df', but with the values in each column discretized using 'fun'.
    """
    # Aplicar la función a todas las columnas
    df_discretized = df.apply(lambda x: fun(x,num_bins)[1])
    return df_discretized

def normalize_data(v):
    """
    This function normalizes a Pandas Series, DataFrame column, or NumPy array.
    
    If the input is a numerical column or array, the function scales the values to the range [0, 1].
    If the input is a categorical column, the function returns the original values unchanged.
    
    Parameters:
        v (pandas.Series, pandas.DataFrame, or numpy.ndarray): The Series, DataFrame column, or NumPy array to be normalized.
    
    Returns:
        pandas.Series or numpy.ndarray: 
            If the input is a Pandas Series or DataFrame column, the function returns a Series with the same index as 'v',
            but with the values normalized.
            If the input is a NumPy array, the function returns a NumPy array with the same shape as 'v', but with the values normalized.
    """
    if v.dtype in ['int64', 'float64']:
        minimo = v.min()
        maximo = v.max()
        return (v - minimo) / (maximo - minimo)
    else:
        return v

def normalize_dataframe(df):
    """
    This function normalizes all numerical columns in a Pandas DataFrame.
    
    The function applies the 'normalize_data' function to each numerical column in the DataFrame,
    and returns a new DataFrame with the normalized values. Categorical columns are left unchanged.
    
    Parameters:
        df (pandas.DataFrame): The DataFrame to be normalized.
    
    Returns:
        pandas.DataFrame: A DataFrame with the same structure as 'df', but with the numerical columns normalized.
    """
    norm_df = df.apply(lambda x: normalize_data(x))
    return norm_df

def standardize(v):
    """
    This function standardizes a Pandas Series, DataFrame column, or NumPy array.
    
    If the input is a numerical column or array, the function scales the values to have zero mean and unit variance.
    If the input is a categorical column, the function returns the original values unchanged.
    
    Parameters:
        v (pandas.Series, pandas.DataFrame, or numpy.ndarray): The Series, DataFrame column, or NumPy array to be standardized.
    
    Returns:
        pandas.Series or numpy.ndarray: 
            If the input is a Pandas Series or DataFrame column, the function returns a Series with the same index as 'v',
            but with the values standardized.
            If the input is a NumPy array, the function returns a NumPy array with the same shape as 'v', but with the values standardized.
    """
    if v.dtype in ['int64', 'float64']:
        return (v - v.mean()) / v.std()
    else:
        return v

def standardize_dataframe(df):
    """
    This function standardizes all numerical columns in a Pandas DataFrame.
    
    The function applies the 'standardize' function to each numerical column in the DataFrame,
    and returns a new DataFrame with the standardized values. Categorical columns are left unchanged.
    
    Parameters:
        df (pandas.DataFrame): The DataFrame to be standardized.
    
    Returns:
        pandas.DataFrame: A DataFrame with the same structure as 'df', but with the numerical columns standardized.
    """
    std_df = df.apply(lambda x: standardize(x))
    return std_df

def calc_entropy(x):
    """
    This function calculates the Shannon entropy of a discrete random variable.
    
    The Shannon entropy is a measure of the uncertainty or randomness of a discrete random variable.
    It is defined as the expected value of the information contained in a message, in bits.
    
    Parameters:
        x (sequence): A sequence of discrete values representing the random variable.
    
    Returns:
        float: The Shannon entropy of the random variable, in bits.
    """
    x = np.array(x)
    unique_elements, counts = np.unique(x, return_counts=True)
    probs = counts / len(x)
    return -np.sum(probs * np.log2(probs))

def calc_variance(x):
    """
    This function calculates the variance of a sample.
    
    The variance is a measure of the spread of the data around the mean.
    
    Parameters:
        x (sequence): A sequence of numerical values representing the sample.
    
    Returns:
        float: The variance of the sample.
    """
    mean = np.mean(x)
    variance = np.sum((x - mean)**2)
    variance /= (len(x))
    return variance

def calc_AUC(df, atributo, labels="label"):
    """
    This function calculates the Area Under the Curve (AUC) for a binary classifier.
    
    The function first calculates the True Positive Rate (TPR) and False Positive Rate (FPR) for a range of attribute values.
    It then plots the resulting TPR and FPR values on a graph and calculates the AUC using an optimized integration method.
    
    Classification is performed by taking the attribute values as thresholds and predicting "True" if the attribute value is bigger and "False" if it is smaller.
    
    Parameters:
        df (pandas.DataFrame): A DataFrame containing the data to be used for the AUC calculation.
        atributo (str): The name of the column in 'df' containing the attribute to be used for the AUC calculation.
        labels (str): The name of the column in 'df' containing the class labels. Default: 'label'.
    
    Returns:
        float: The AUC value for the binary classifier.
    """
    valores = np.array(df[atributo])
    valores = np.sort(valores)
    TPR = []
    FPR = []
    for corte in valores:
        t,f = TPR_FPR(df, corte, atributo, labels)
        TPR.append(t)
        FPR.append(f)
    TPR.append(0)
    FPR.append(0)    
    AUC = abs(integraOptimizada(FPR, TPR))
    plot_auc(FPR, TPR, AUC)
    return AUC

def TPR_FPR(df, corte, atributo, labels):
    """
    This function calculates the True Positive Rate (TPR) and False Positive Rate (FPR) for a binary classifier.
    
    The function takes a DataFrame, a threshold value, and the names of an attribute column and a labels column as input.
    It returns the TPR and FPR for the classifier using the attribute values as thresholds to make predictions and compare them to the labels.
    
    Parameters:
        df (pandas.DataFrame): A DataFrame containing the data to be used for the TPR and FPR calculation.
        corte (float): The threshold value to be used for the prediction.
        atributo (str): The name of the column in 'df' containing the attribute to be used for the prediction.
        labels (str): The name of the column in 'df' containing the class labels.
    
    Returns:
        tuple: A tuple containing the TPR and FPR values.
    """
    
    TN = df.loc[(df[atributo]<corte) & (df[labels]==False)]
    TN = TN.shape[0]

    FN = df.loc[(df[atributo]<corte) & (df[labels]==True)]
    FN = FN.shape[0]

    TP = df.loc[(df[atributo]>=corte) & (df[labels]==True)]
    TP = TP.shape[0]

    FP = df.loc[(df[atributo]>=corte) & (df[labels]==False)]
    FP = FP.shape[0]

    TPR = TP/(TP+FN)
    FPR = FP/(FP+TN)
    return TPR, FPR

def checkboolean(label_col):
    """
    This function checks if a Pandas Series contains boolean or binary values, and converts it to boolean if necessary.
    
    The function takes a Pandas Series as input and checks if its dtype is boolean. If it is, the function does nothing.
    If the dtype is numeric, the function checks if the Series contains only two unique values, and converts it to boolean if it does.
    If the Series is neither boolean nor binary, the function raises a ValueError.
    
    Parameters:
        label_col (pandas.Series): The Pandas Series to be checked and possibly converted to boolean.
    
    Returns:
        pandas.Series: A Pandas Series with the same values as 'label_col', but with a boolean dtype.
    
    Raises:
        ValueError: If 'label_col' is neither boolean nor binary.
    """
    if label_col.dtype == 'bool':
    # No hacer nada si es boolean
        pass
    elif label_col.dtype in ['int64', 'float64']:
        # Mirar si es binario
        if len(label_col.unique()) == 2:
            # si es binario, convertir a boolean
            label_col = label_col.astype(bool)
        else:
            # Si no es binario ni boolean, mandar error
            raise ValueError("Column is not binary.")
    else:
        # Si no es boolean ni numérico, mandar error
        raise ValueError("Column is not boolean or numeric.")

def integraOptimizada(x, y):
    """
    This function calculates the area under a curve using an optimized integration method.
    
    The function takes two numpy arrays 'x' and 'y' as input, and calculates the area under the curve formed by the points (x[i], y[i]).
    The function returns the result of the integration.
    
    Parameters:
        x (numpy.array): A numpy array containing the x-values of the points to be integrated.
        y (numpy.array): A numpy array containing the y-values of the points to be integrated.
    
    Returns:
        float: The result of the integration.
    """
    delta_x = np.diff(x)
    a=np.delete(y,0) #Para quietar los 0s del principio 
    b=np.delete(y,len(y)-1) # y del final
    mean_y=np.mean(np.column_stack((a,b)), axis=1)
    return(np.sum(np.dot(delta_x,mean_y)))

def calc_metrics_df(df, labels="No labels"):
    """
    This function calculates metrics for all columns in a Pandas DataFrame.
    
    The function takes a DataFrame as input, and calculates the entropy for discrete columns and variance for continuous columns.
    If a labels column is provided, the function calculates the Area Under the Curve (AUC) for all continuous columns.
    The function returns a DataFrame containing the metrics for each column.
    
    Parameters:
        df (pandas.DataFrame): The DataFrame for which the metrics are to be calculated.
        labels (str, optional): The name of the column in 'df' containing the class labels. Defaults to "No labels".
    
    Returns:
        pandas.DataFrame: A DataFrame containing the metrics for each column in 'df'.
    """
    metric_df = pd.DataFrame(columns=['Column', 'Entropy', 'Variance', 'AUC'])
    if labels!="No labels":
        checkboolean(df[labels])
    
    for col in df.columns:
        if df[col].dtype == 'int64':
            entropy = calc_entropy(df[col])
            variance = None
            auc = None
            
        elif df[col].dtype == 'float64':
            entropy = None
            variance = calc_variance(df[col])
            if labels!="No labels":
                auc = calc_AUC(df, col, labels)
            else:
                auc = None
                
        #Añadir métricas de la columna al dataframe de la respuesta
        metric_df = metric_df.append({'Column': col, 'Entropy': entropy, 'Variance': variance, 'AUC': auc}, ignore_index=True)
    return metric_df

def filter_df(df, condition, labels="No labels"):
    """
    This function filters columns in a Pandas DataFrame based on a given condition.
    
    The function takes a DataFrame and a string containing a condition in the form "metric operator number", such as "Entropy > 0.5".
    The function returns the names of the columns in 'df' that satisfy the given condition.
    
    Parameters:
        df (pandas.DataFrame): The DataFrame to be filtered.
        condition (str): The string containing the condition to be used for filtering.
        labels (str, optional): The name of the column in 'df' containing the class labels. Defaults to "No labels".
    
    Returns:
        pandas.Dataframe: A Pandas Dataframe containing the columns in 'df' that satisfy the given condition.
    """
    metric_df = calc_metrics_df(df, labels=labels)
    try:
        for i, char in enumerate(condition):
            if char == '<':
                if condition[i+1] == "=":
                    metric = condition[:i]
                    num = condition[i+2:]
                    num = float(num)
                    filtered = metric_df.loc[metric_df[metric]<=num, "Column"]
                    break
                else:
                    metric = condition[:i]
                    num = condition[i+1:]
                    num = float(num)
                    filtered = metric_df.loc[metric_df[metric]<num, "Column"]
                    break   
            if char == '>':
                if condition[i+1] == "=":
                    metric = condition[:i]
                    num = condition[i+2:]
                    num = float(num)
                    filtered = metric_df.loc[metric_df[metric]>=num, "Column"]
                    break
                else:
                    metric = condition[:i]
                    num = condition[i+1:]
                    num = float(num)
                    filtered = metric_df.loc[metric_df[metric]>num, "Column"]
                    break  
            if char == '=':
                metric = condition[:i]
                num = condition[i+2:]
                num = float(num)
                filtered = metric_df.loc[metric_df[metric]==num, "Column"]
                break
            if char == '!':
                metric = condition[:i]
                num = condition[i+2:]
                num = float(num)
                filtered = metric_df.loc[metric_df[metric]!=num, "Column"]
                break
    except:
        raise ValueError("Invalid input")
        pass
    
    
    filtered = filtered.to_numpy()
    df = df.loc[:, filtered]
    return df

def calc_correlation(x, y):
    """
    This function calculates the Pearson correlation coefficient between two variables.
    
    It takes two NumPy arrays or columns in a Pandas DataFrame and returns the Pearson correlation coefficient.
    
    Parameters:
        x (np.array or pd.Series): The first variable.
        y (np.array or pd.Series): The second variable.
    
    Returns:
        float: The Pearson correlation coefficient between 'x' and 'y'.
    """
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_std = np.std(x)
    y_std = np.std(y)
    cov = np.mean((x - x_mean) * (y - y_mean))
    return cov / (x_std * y_std)

def calc_MI(X,Y,n_bins):
    """
    This function calculates the mutual information between two variables.
    
    The function takes two NumPy arrays or columns in a Pandas DataFrame and the number of bins to use when calculating the mutual information. It returns the mutual information between the two variables.
    
    Parameters:
        X (np.array or pd.Series): The first variable.
        Y (np.array or pd.Series): The second variable.
        n_bins (int): The number of bins to use when calculating the mutual information.
    
    Returns:
        float: The mutual information between 'X' and 'Y'.
    """
    XY_his = np.histogram2d(X,Y,n_bins)[0]
    X_his = np.histogram(X,n_bins)[0]
    Y_his = np.histogram(Y,n_bins)[0]
    
    #Calcular las entropías
    XY_H = calc_entropy(XY_his)
    X_H = calc_entropy(X_his)
    Y_H = calc_entropy(Y_his)
    
    MI = X_H + Y_H - XY_H
    return MI

def normalized_entropy_2(x):
    """Calculate the normalized entropy of a sequence x.
    
    The normalized entropy of a sequence is defined as the entropy of the sequence
    divided by the maximum possible entropy for a sequence of the same length.
    This helps to compare the randomness of sequences of different lengths.
    
    Parameters:
        x (sequence): The input sequence.
    
    Returns:
        float: The normalized entropy of x.
    """
    x = np.array(x)
    unique_vals, counts = np.unique(x, return_counts=True)
    probs = counts / len(x)
    log_probs = np.log2(probs)
    entropy = -np.sum(log_probs)
    return entropy / np.log2(len(unique_vals))

def metricas_pares(df):
    """
    Calculate mutual information (MI) or correlation between all pairs of columns in a dataframe. 
    If one of the columns is discrete (categorical factorized to integers), mutual information is calculated. 
    If both columns are continuous, correlation is calculated.

    Parameters:
        df (pandas.DataFrame): The input dataframe.

    Returns:
        tuple: Two dataframes containing the MI and correlation values, respectively.
    """
    num_cols = len(df.columns)
    results_MI = np.zeros((num_cols, num_cols))
    results_cor = np.zeros((num_cols, num_cols))


    # Iterar a través de todos los pares de columnas
    for i, col1 in enumerate(df.columns):
        for j, col2 in enumerate(df.columns):
            if i < j:  
                if df[col1].dtype =='int64' or df[col2].dtype =='int64':
                    results_MI[i,j] = calc_MI(df[col1] , df[col2], 5)
                else:
                    results_cor[i, j] = calc_correlation(df[col1] , df[col2])

    i, j = np.triu_indices(num_cols, k=1)
    results_MI[j, i] = results_MI[i, j]
    results_cor[j, i] = results_cor[i, j]
    
    df_results_MI = filter_results(results_MI, df.columns)
    df_results_cor = filter_results(results_cor, df.columns)

    return df_results_MI, df_results_cor

def filter_results(matrix, colnames):
    """
    Filter rows and columns containing only zeros from the input matrix and return a dataframe with the resulting values.

    Parameters:
        matrix (numpy.ndarray): The input matrix.
        colnames (list): A list of column names corresponding to the columns in the input matrix.

    Returns:
        pandas.DataFrame: The filtered dataframe.
    """
    #Se eliminan las filas y columnas que contengan solo 0s
    indices = np.where(matrix.any(axis=1))[0]
    matrix_filtered = matrix[indices[:, None], indices]
    df_results = pd.DataFrame(matrix_filtered, columns=colnames[indices], index=colnames[indices])
    return df_results

def plot_MI(res1):
    res1_nozero = res1.replace(0, np.nan)
    vmin = res1_nozero.min().min()-1
    vmax = res1_nozero.max().max()+1
    res1 = res1.round(decimals=2)
    sns.heatmap(res1, annot=True, fmt='g', vmin=vmin, vmax=vmax)

def plot_cor(res2):
    res2 = res2.round(decimals=2)
    vmin = res2.min().min()-0.3
    vmax = res2.max().max()+0.3
    sns.heatmap(res2, annot = True, vmin=vmin, vmax=vmax)

def metricas_pares(df, heatmaps=True):
    """
    Calculate mutual information (MI) or correlation between all pairs of columns in a dataframe. 
    If one of the columns is discrete (categorical factorized to integers), mutual information is calculated. 
    If both columns are continuous, correlation is calculated.
    There is an option to add heatmaps of the results.

    Parameters:
        df (pandas.DataFrame): The input dataframe.
        heatmaps (boolean): Boolean value to indicate if it is necessary to show heatmaps of the results. Defaults True.

    Returns:
        tuple: Two dataframes containing the MI and correlation values, respectively.
    """
    num_cols = len(df.columns)
    results_MI = np.zeros((num_cols, num_cols))
    results_cor = np.zeros((num_cols, num_cols))


    # Iterar a través de todos los pares de columnas
    for i, col1 in enumerate(df.columns):
        for j, col2 in enumerate(df.columns):
            if i < j:  
                if df[col1].dtype =='int64' or df[col2].dtype =='int64':
                    results_MI[i,j] = calc_MI(df[col1] , df[col2], 5)
                else:
                    results_cor[i, j] = calc_correlation(df[col1] , df[col2])

    i, j = np.triu_indices(num_cols, k=1)
    results_MI[j, i] = results_MI[i, j]
    results_cor[j, i] = results_cor[i, j]
    
    df_results_MI = filter_results(results_MI, df.columns)
    df_results_cor = filter_results(results_cor, df.columns)
    
    if heatmaps:
        #Heatmap de información mutua
        ax1 = plt.axes()
        res1_nozero = df_results_MI.replace(0, np.nan)
        vmin = res1_nozero.min().min()-1
        vmax = res1_nozero.max().max()+1
        df_results_MI = df_results_MI.round(decimals=2)
        sns.heatmap(df_results_MI, annot=True, fmt='g', vmin=vmin, vmax=vmax)
        ax1.set_title('Mapa de calor de informaciones mutuas')
        plt.show()
        
        #Heatmap de correlación
        ax2 = plt.axes()
        res2 = df_results_cor.round(decimals=2)
        vmin = res2.min().min()-0.3
        vmax = res2.max().max()+0.3
        sns.heatmap(res2, annot = True, vmin=vmin, vmax=vmax)
        ax2.set_title('Mapa de calor de correlaciones')
        plt.show()
    return df_results_MI, df_results_cor

def plot_auc(fpr, tpr, auc):
    """
    Function to plot the ROC curve using plotly and the false positive rate (fpr) and true positive rate (tpr) arrays from the function 'calc_auc',it also displays the AUC.

    Parameters:
        fpr (list): A list of false positive rates.
        tpr (list): A list of true positive rates.
        auc (float): The area under the curve.

    Returns:
        None
    """

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=fpr, y=tpr,
                        mode='lines+markers'))
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1,
    )
    
    fig.update_layout(
        xaxis=dict(range=[-0.009, 1.009]), 
        yaxis=dict(range=[-0.009, 1.009]),
        title=f'ROC Curve (AUC={auc:.4f})',
    )
    
    fig.update_layout(
        xaxis=dict(title='False Positive Rate'),  # Add a label to the x-axis
        yaxis=dict(title='True Positive Rate')   # Add a label to the y-axis
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    fig.show()        

