import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K



# import shutil
# !git clone https://github.com/google-research/fast-soft-sort.git
# try:
#   shutil.move("fast-soft-sort/fast_soft_sort",".")
# except:
#   pass
# finally:
#   shutil.rmtree("fast-soft-sort")
  
# #Use this function to do sorting part in spearman correlation...By doing this you should be able to create a differentiable speramans loss function
# #Spearman is preferred for numerai

# from fast_soft_sort import numpy_ops
# import tensorflow as tf

# def _wrap_numpy_op(cls, regularization_strength, direction, regularization):
#   """Converts NumPy operator to a TF one."""

#   @tf.custom_gradient
#   def _func(values):
#     """Converts values to numpy array, applies function and returns tensor."""
#     dtype = values.dtype

#     try:
#       values = values.numpy()
#     except AttributeError:
#       pass

#     obj = cls(values, regularization_strength=regularization_strength,
#               direction=direction, regularization=regularization)
#     result = obj.compute()

#     def grad(v):
#       v = v.numpy()
#       return tf.convert_to_tensor(obj.vjp(v), dtype=dtype)

#     return tf.convert_to_tensor(result, dtype=dtype), grad

#   return _func


# def soft_rank(values, direction="ASCENDING", regularization_strength=1.0,
#               regularization="l2"):
#   r"""Soft rank the given values (tensor) along the second axis.

#   The regularization strength determines how close are the returned values
#   to the actual ranks.

#   Args:
#     values: A 2d-tensor holding the numbers to be ranked.
#     direction: Either 'ASCENDING' or 'DESCENDING'.
#     regularization_strength: The regularization strength to be used. The smaller
#     this number, the closer the values to the true ranks.
#     regularization: Which regularization method to use. It
#       must be set to one of ("l2", "kl", "log_kl").
#   Returns:
#     A 2d-tensor, soft-ranked along the second axis.
#   """
#   if len(values.shape) != 2:
#     raise ValueError("'values' should be a 2d-tensor "
#                      "but got %r." % values.shape)

#   assert tf.executing_eagerly()

#   func = _wrap_numpy_op(numpy_ops.SoftRank, regularization_strength, direction,
#                         regularization)

#   return tf.map_fn(func, values)



# values = tf.convert_to_tensor([[5., 1., 2.]], dtype=tf.float64)
# tf.print(tf.argsort(tf.argsort(values)))
# soft_rank(values, regularization_strength=0.01) - 1
    
#Spearman Correlation
#Not giving same results as tf_correlation
#values about 100 times higher than tf_correlation in training phase...
class CustomCorr(tf.keras.metrics.Metric):
  def __init__(self,name="corr",**kwargs):
    super().__init__(name=name,**kwargs)
    self.corr = self.add_weight(name="clp",initializer='zeros')

  def update_state(self,targets,predictions,sample_weight=None):
    predictions = tf.squeeze(predictions) 
    targets = tf.squeeze(targets)
    ranks = tf.argsort(tf.argsort(predictions,stable=True))
    #tf.print(ranks)
    #print(K.shape(ranks))
    ranked_preds = tf.cast(tf.math.divide(ranks,K.shape(ranks)), targets.dtype)
    self.corr.assign_add(self.corrcoef(ranked_preds,targets))

  @staticmethod
  def corrcoef(x, y):
    """
    np.corrcoef() implemented with tf primitives
    """
    mx = tf.math.reduce_mean(x)
    my = tf.math.reduce_mean(y)
    xm, ym = x - mx, y - my
    r_num = tf.math.reduce_sum(xm * ym)
    r_den = tf.norm(xm) * tf.norm(ym)
    return r_num / (r_den + tf.keras.backend.epsilon())

  def result(self):
    return self.corr

  def reset_states(self):
    self.corr.assign(0.0)


#Not Giving the same correlation values  as normal implementation, rejecting for now
#This is a differentiable implementation of Spearmans, could be used as a loss function
#Need to use run_eagerly=True in model.compile when running this...

# class CustomCorrFast(tf.keras.metrics.Metric):
#   def __init__(self,name="corr_fast",**kwargs):
#     super().__init__(name=name,**kwargs)
#     self.corr = self.add_weight(name="clp",initializer='zeros')

#   def update_state(self,y,x,sample_weight=None):
#     #tf.print(tf.transpose(x))
#     ranked_x = tf.cast(tf.math.subtract(soft_rank(tf.transpose(x), regularization_strength=1),1), y.dtype)
#     #tf.print(tf.transpose(ranked_x))
#     self.corr.assign_add(self.corrcoef(ranked_x,y))

#   @staticmethod
#   def corrcoef(x, y):
#     """
#     np.corrcoef() implemented with tf primitives
#     """
#     mx = tf.math.reduce_mean(x)
#     my = tf.math.reduce_mean(y)
#     xm, ym = x - mx, y - my
#     r_num = tf.math.reduce_sum(xm * ym)
#     r_den = tf.norm(xm) * tf.norm(ym)
#     return r_num / (r_den + tf.keras.backend.epsilon())

#   def result(self):
#     return self.corr

#   def reset_states(self):
#     self.corr.assign(0.0)
