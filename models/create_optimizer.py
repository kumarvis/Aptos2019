import tensorflow as tf
from .tf_rectified_adam import RectifiedAdam
from config.img_classification_config import ConfigObj

def create_optimizer(learning_rate):
    # Setup optimizer
    if ConfigObj.optimizer == "adadelta":
      optimizer = tf.train.AdadeltaOptimizer(
          learning_rate=learning_rate)
    elif ConfigObj.optimizer == "adagrad":
      optimizer = tf.train.AdagradOptimizer(
          learning_rate=learning_rate)
    elif ConfigObj.optimizer == "adam":
      optimizer = tf.train.AdamOptimizer(
          learning_rate=learning_rate)
    elif ConfigObj.optimizer == "radam":
      optimizer = RectifiedAdam(
          learning_rate=learning_rate)
    elif ConfigObj.optimizer == "ftrl":
      optimizer = tf.train.FtrlOptimizer(
          learning_rate=learning_rate)
    elif ConfigObj.optimizer == "momentum":
      optimizer = tf.train.MomentumOptimizer(
          learning_rate=learning_rate,
          momentum=0.9,
          name="Momentum")
    elif ConfigObj.optimizer == "rmsprop":
      optimizer = tf.train.RMSPropOptimizer(
          learning_rate=learning_rate)
    elif ConfigObj.optimizer == "sgd":
      optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate)
    else:
      raise ValueError("Optimizer [%s] was not recognized" %
                       ConfigObj.optimizer)
    return optimizer