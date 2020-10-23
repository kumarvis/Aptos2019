import os
import tensorflow as tf
from src.img_classification_csv_input import ImageClassificationCSVDataPipeline
from src.config import ConfigObj
from src.prepare_model import get_custom_model
import math
from optimal_lr_finder_src.LRFinder import LRFinder

data_pipeline_obj = ImageClassificationCSVDataPipeline()
dataset = data_pipeline_obj.get_tf_dataset()
train_ds, valid_ds = data_pipeline_obj.split_dataset(dataset, ConfigObj.Validation_Fraction)
custom_model = get_custom_model()

## LR Finder
min_lr, max_lr = 0.00001, 0.1
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
log_file_path = os.path.join(ConfigObj.Path_Parent_Dir, 'optimal_lr_finder_src', 'optimal_lr.log') 

lr_finder = LRFinder(custom_model, optimizer, loss_fn, train_ds)
lr_finder.range_test(min_lr, max_lr)
lr_finder.dump(log_file_path)
lr_finder.plot_smooth_curve(log_file_path)

print('----> EXPERIMENT FINISHED <----')


