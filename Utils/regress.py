import pandas as pd
import numpy as np
from statsmodels import regression
from statsmodels.tools import add_constant
from scipy import stats
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)


def regression_calcs(df: pd.DataFrame,
                     y: np.ndarray,
                     r2: float = None,
                     constant_coef: float = None,
                     variable_coef: float = None,
                     confidence_interval_lower: float = None,
                     confidence_interval_higher: float = None,
                     p_value: float = 0.05,
                     sort_value: list = None,
                     filtering: bool = False,
                     ):
    if sort_value is None:
        sort_value = ['P Value', True]
    
    final_cols = []
    if r2 is not None:
        final_cols.append('R Squared')
    
    if constant_coef is not None:
        final_cols.append('Constant Coefficient')
    
    if variable_coef is not None:
        final_cols.append('Word Coefficient')
    
    if confidence_interval_lower is not None:
        final_cols.append('Lower Confidence Interval')
    
    if confidence_interval_higher is not None:
        final_cols.append('Higher Confidence Interval')
    
    if p_value is not None:
        final_cols.append('P Value')
    
    col_lst = list(df.columns)
    result = {}
    for word in col_lst:
        x = add_constant(np.array(df[word], dtype=float))
        model = regression.linear_model.OLS(y, x).fit()
        
        model_results = []
        if r2 is not None:
            model_results.append(model.rsquared)
        
        if constant_coef is not None:
            model_results.append(model.params[0])
        
        if variable_coef is not None:
            model_results.append(model.params[1])
        
        if confidence_interval_lower is not None:
            model_results.append(model.conf_int()[1, :2][0])
        
        if confidence_interval_higher is not None:
            model_results.append(model.conf_int()[1, :2][1])
        
        if p_value is not None:
            model_results.append(model.pvalues[1])
        
        result[word] = model_results
    
    result_df = pd.DataFrame.from_dict(result, orient='index', columns=final_cols)
    if filtering:
        if r2 is not None:
            result_df = result_df[result_df['R Squared'] > r2]
        
        if constant_coef is not None:
            result_df = result_df[result_df['Constant Coefficient'] > constant_coef]
        
        if variable_coef is not None:
            result_df = result_df[result_df['Word Coefficient'] > variable_coef]
        
        if confidence_interval_lower is not None:
            result_df = result_df[result_df['Lower Confidence Interval'] > confidence_interval_lower]
        
        if confidence_interval_higher is not None:
            result_df = result_df[result_df['Higher Confidence Interval'] > confidence_interval_higher]
        
        if p_value is not None:
            result_df = result_df[result_df['P Value'] < p_value]
    
    final = result_df.sort_values(sort_value[0], ascending=sort_value[1]).round(3)
    
    return final, list(final.index)

def regress(xtrain,
            ytrain,
            xtest,
            ytest,
            test=False,
            plotn=False
            ):
    
        X = add_constant(xtrain)
        result = np.linalg.lstsq(X, ytrain, rcond=None)
        alpha, beta = result[0]
    
        # coef = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(ytrain)
        r2 = np.corrcoef(xtrain, ytrain)[0, 1] ** 2  # r^2
    
        y_hat = beta * xtrain + alpha
        res = ytrain - y_hat
        var_beta_hat = np.linalg.inv(X.T @ X) * ((res.T @ res) / (len(xtrain) - 2))
        std_error = tuple(np.diag(var_beta_hat) ** .5)
        t, p = stats.ttest_ind(ytrain, xtrain, equal_var=False)  # t test and p value
        p_m = stats.norm.ppf(.95) * (np.std(xtrain) / np.sqrt(len(xtrain)))
        conf = (np.mean(xtrain) - p_m, np.mean(xtrain) + p_m)
    
        x_fit = np.linspace(np.floor(xtrain.min()), np.ceil(xtrain.max()), 2)
        y_fit = alpha * x_fit + beta
    
        if plotn:
            plt.scatter(xtrain, ytrain, marker='o')
            plt.plot(x_fit, y_fit, 'r--')
            plt.title('Training Best-fit linear model')
            plt.show()
    
        if test:
            pred = np.dot(add_constant(xtest), result[0])
            resid = np.dot(add_constant(xtest), result[0]) - ytest
            mse = np.mean(np.square(ytest - xtest))
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(xtest - ytest))
        
            if plotn:
                plt.scatter(pred, resid, marker='o')
                plt.title('Test Best-fit linear model')
                plt.show()
        
            print('Intercept, Beta, R2, Standard Error, T-Statistic, P-Values, Confidence Inter')
            print('MSE, RMSE, MAE')
            print('Predicted, Residuals')
            return (((alpha, beta), r2, std_error, t, p, conf), (mse, rmse, mae), (pred, resid))
    
        else:
            print('Intercept, Beta, R2, Standard Error, T-Statistic, P-Values, Confidence Inter')
            return ((alpha, beta), r2, std_error, t, p, conf)