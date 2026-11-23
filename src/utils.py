import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score as r2, mean_squared_error as mse, accuracy_score, precision_score, f1_score, recall_score, roc_auc_score, roc_curve, classification_report

def evaluate_preds(true_values, pred_values, save: bool=False):
    true_values = np.ravel(true_values)
    pred_values = np.ravel(pred_values)

    fig, ax = plt.subplots()
    textstr = ("R2:  " + str(round(r2(true_values, pred_values), 3)) + "\n" +
          "RMSE:  " + str(round(np.sqrt(mse(true_values, pred_values)), 3)) + "\n" +
          "MSE:  " + str(round(mse(true_values, pred_values), 3)))
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)    
    sns.scatterplot(x=pred_values, y=true_values)
    ax.plot([min(pred_values), max(pred_values)], [min(true_values), max(true_values)], linestyle='--', color='black')
    plt.xlabel('Predicted values')
    plt.ylabel('True values')
    plt.title('True vs Predicted values')

    plt.show()

    return round(r2(true_values, pred_values), 3), round(np.sqrt(mse(true_values, pred_values)), 3)

def show_metrics(y_test, y_pred, y_pred_proba=None, average='binary'):
      print(classification_report(y_test, y_pred, digits=4))
      if (y_pred_proba is not None) and (average == 'binary'):
            print(f'roc_auc_score = {roc_auc_score(y_test, y_pred_proba)}')
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=1)
            plt.rcParams['figure.figsize'] = 5, 5

            plt.plot(fpr, tpr)
            plt.plot([0, 1], [0, 1], color='grey', linestyle='dashed')

            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC curve')

            plt.show()

def show_weights(pipeline):
      model = pipeline.named_steps['log_reg']
      importances = model.coef_[0]
      feature_names = model.feature_names_in_

      for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
            print(f"{name}: {imp:.4f}")