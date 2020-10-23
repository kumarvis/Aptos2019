import os
import tensorflow as tf
from src_base.img_classification_csv_input import ImageClassificationCSVDataPipeline
from config.img_classification_config import ConfigObj
from src_base.prepare_model import get_custom_model
import math
from src_optimal_lr_finder.LRFinder import LRFinder

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


