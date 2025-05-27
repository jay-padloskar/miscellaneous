import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, FunctionTransformer, RobustScaler, KBinsDiscretizer, StandardScaler
import optuna
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from tqdm import tqdm
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from catboost import CatBoostClassifier, Pool, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import VotingClassifier, StackingClassifier, StackingRegressor
from sklearn.metrics import root_mean_squared_log_error, root_mean_squared_error, mean_squared_error, accuracy_score, roc_auc_score, classification_report, f1_score, precision_score, recall_score, confusion_matrix, mean_absolute_percentage_error, matthews_corrcoef
from scipy.stats import mode
import plotly.express as px
import missingno as msno
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVC, SVR
from sklearn.impute import KNNImputer
from sklearn.feature_selection import mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from yellowbrick.features import FeatureImportances
from yellowbrick.classifier import ConfusionMatrix, ClassificationReport, ROCAUC, DiscriminationThreshold
import shap
from itertools import combinations
from IPython.display import display

sns.set(style = 'dark', palette = 'terrain', font_scale = 1.2)

print('\nImporting Libraries is a Success!')
