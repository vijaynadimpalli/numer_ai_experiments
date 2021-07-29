import scipy
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import tensorflow.keras.backend as K
import xgboost as xgb

TARGET_NAME = f"target"
PREDICTION_NAME = f"prediction"

#Spearman Correlation
#NOTE: Targets before predictions in arguments, might be different for your model.
def correlation(targets, predictions):
    """
    From:
    https://github.com/numerai/example-scripts/blob/master/example_model.py#L21
    """    
    if not isinstance(predictions, pd.Series):
        predictions = pd.Series(predictions)
    ranked_preds = predictions.rank(pct=True, method="first")
    return np.corrcoef(ranked_preds, targets)[0, 1]


# convenience method for scoring
def score(df):
    return correlation(df[TARGET_NAME], df[PREDICTION_NAME])

def correlation_xgboost_wrapper(predictions, dTrain):

    targets = dTrain.get_label()
    return 'corr', correlation(targets, predictions)


# Payout is just the score cliped at +/-25%
def payout(scores):
    return scores.clip(lower=-0.25, upper=0.25)

# Remove the regression portion 
def _neutralize(df, columns, by, proportion=1.0):
    scores = df[columns]
    exposures = df[by].values
    scores = scores - proportion * exposures.dot(np.linalg.pinv(exposures).dot(scores))
    return scores / scores.std(ddof=0)

def _normalize(df):
    X = (df.rank(method="first") - 0.5) / len(df)
    return scipy.stats.norm.ppf(X)

def normalize_and_neutralize(df, columns, by, proportion=1.0):
    # Convert the scores to a normal distribution
    df[columns] = _normalize(df[columns])
    df[columns] = _neutralize(df, columns, by, proportion)
    return df[columns]

#Calculating feature exposures
def feature_exposures(df,features,pred_name='preds'):
    if not isinstance(df,pd.DataFrame):
      raise Exception("Not a DataFrame")
    feat_expo = []
    for feature in features:
        feat_expo.append(np.corrcoef(df[feature], df[pred_name])[0,1])
    feat_expo = pd.Series(feat_expo, index=features)
    return feat_expo


# metric to account for auto correlation
def autocorr_penalty(x):
    n = len(x)
    p = np.abs(np.corrcoef(x[:-1], x[1:])[0,1])
    return np.sqrt(1 + 2*np.sum([((n - i)/n)*p**i for i in range(1,n)]))

# penalize sharpe with auto correlation
def smart_sharpe(x):
    return np.mean(x)/(np.std(x, ddof=1)*autocorr_penalty(x))

# penalize sortino with auto correlation
def smart_sortino_ratio(x, target=.02):
    xt = x - target
    return np.mean(xt)/(((np.sum(np.minimum(0, xt)**2)/(len(xt)-1))**.5)*autocorr_penalty(x))



def tf_corrcoef(x, y):
  """
  np.corrcoef() implemented with tf primitives
  """
  mx = tf.math.reduce_mean(x)
  my = tf.math.reduce_mean(y)
  xm, ym = x - mx, y - my
  r_num = tf.math.reduce_sum(xm * ym)
  r_den = tf.norm(xm) * tf.norm(ym)
  return r_num / (r_den + tf.keras.backend.epsilon())

#Spearman TF Implementation
#Giving same results as get_spearman_rankcor
def tf_correlation(targets, predictions):
    #Squeezing to remove dimensions of size 1
    predictions = tf.squeeze(predictions) 
    targets = tf.squeeze(targets)

    ranks = tf.argsort(tf.argsort(predictions,stable=True))
    ranked_preds = tf.cast(tf.math.divide(ranks,K.shape(ranks)), targets.dtype)
    #Dividing by length to convert range of ranks to (0,1)
    #Using K.shape instead of tf.shape, since using tf.shape produced an error (could not give exact batch size, as shape of 1st dim is None)

    return tf_corrcoef(ranked_preds, targets)


def get_spearman_rankcor(y_true, y_pred):
     return (tf.numpy_function(correlation, [tf.squeeze(tf.cast(y_true, tf.float32)), tf.squeeze(tf.cast(y_pred, tf.float32))], Tout = tf.float64) )
     

def test():
    targets = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]], dtype=np.float32)
    predictions = np.array([[0.583], [0.527] ,[0.493], [0.425], [0.485]], dtype=np.float32)
    print(tf_correlation(tf.convert_to_tensor(targets),tf.convert_to_tensor(predictions)))
    print(get_spearman_rankcor(tf.convert_to_tensor(targets),tf.convert_to_tensor(predictions)))
    
    
   
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args

class TimeSeriesSplitGroups(_BaseKFold):
    def __init__(self, n_splits=5):
        super().__init__(n_splits, shuffle=False, random_state=None)

    def split(self, X, y=None, groups=None):
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        group_list = np.unique(groups)
        n_groups = len(group_list)
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds ={0} greater"
                 " than the number of samples: {1}.").format(n_folds,
                                                             n_groups))
        indices = np.arange(n_samples)
        test_size = (n_groups // n_folds)
        test_starts = range(test_size + n_groups % n_folds,
                            n_groups, test_size)
        test_starts = list(test_starts)[::-1]
        for test_start in test_starts:
            
            yield (indices[groups.isin(group_list[:test_start])],
                   indices[groups.isin(group_list[test_start:test_start + test_size])])


class PurgedGroupTimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator variant with non-overlapping groups.
    Allows for a gap in groups to avoid potentially leaking info from
    train into test if the model has windowed or lag features.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_group_size : int, default=Inf
        Maximum group size for a single training set.
    group_gap : int, default=None
        Gap between train and test
    max_test_group_size : int, default=Inf
        We discard this number of groups from the end of each train split
    """

    @_deprecate_positional_args
    def __init__(self,
                 n_splits=5,
                 *,
                 max_train_group_size=np.inf,
                 max_test_group_size=np.inf,
                 group_gap=None,
                 verbose=False
                 ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_group_size = max_train_group_size
        self.group_gap = group_gap
        self.max_test_group_size = max_test_group_size
        self.verbose = verbose


    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if groups is None:
            raise ValueError(
                "The 'groups' parameter should not be None")
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        group_gap = self.group_gap
        max_test_group_size = self.max_test_group_size
        max_train_group_size = self.max_train_group_size
        n_folds = n_splits + 1
        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds={0} greater than"
                 " the number of groups={1}").format(n_folds,
                                                     n_groups))

        group_test_size = min(n_groups // n_folds, max_test_group_size)
        group_test_starts = range(n_groups - n_splits * group_test_size,
                                  n_groups, group_test_size)
        for group_test_start in group_test_starts:
            train_array = []
            test_array = []

            group_st = max(0, group_test_start - group_gap - max_train_group_size)
            for train_group_idx in unique_groups[group_st:(group_test_start - group_gap)]:
                train_array_tmp = group_dict[train_group_idx]
                
                train_array = np.sort(np.unique(
                                      np.concatenate((train_array,
                                                      train_array_tmp)),
                                      axis=None), axis=None)

            train_end = train_array.size
 
            for test_group_idx in unique_groups[group_test_start:
                                                group_test_start +
                                                group_test_size]:
                test_array_tmp = group_dict[test_group_idx]
                test_array = np.sort(np.unique(
                                              np.concatenate((test_array,
                                                              test_array_tmp)),
                                     axis=None), axis=None)

            test_array  = test_array[group_gap:]
            
            
            if self.verbose > 0:
                    pass
                    
            yield [int(i) for i in train_array], [int(i) for i in test_array]
            


def smooth_mean(array, alpha=0):
    """
    Gives exponentially increasing weights to later folds
    Used in time series based CV
    """    
    if alpha == 0:
        return array.mean()
    else:
        return pd.Series(array).ewm(alpha=alpha).mean().iat[-1]
