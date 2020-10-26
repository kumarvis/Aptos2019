import os
from matplotlib import pyplot as plt
import pathlib
import csv
from pathlib import Path
from config.img_classification_config import ConfigObj

def dumphist2csv(history, prefix):
    plot_path = os.path.join(ConfigObj.Path_Parent_Dir, 'src_train_model', 'plots_logs')

    hist_lbls = ['train_acc', 'val_acc', 'loss', 'val_loss']
    train_accuracy = history.history['acc']
    valid_accuracy = history.history['val_acc']
    train_loss = history.history['loss']
    valid_loss = history.history['val_loss']

    hist_csv_path = os.path.join(plot_path, prefix + "_" + "hist_log.csv")
    with open(hist_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(hist_lbls)
        writer.writerow(train_accuracy)
        writer.writerow(valid_accuracy)
        writer.writerow(train_loss)
        writer.writerow(valid_loss)

def plot_hist_frm_csv(hist_log_path='hist_log.csv'):
    plot_path = os.path.join(ConfigObj.Path_Parent_Dir, 'src_train_model', 'plots_logs')

    with open(hist_log_path) as csvfile:
        read_hist = csv.reader(csvfile, delimiter=',')
        next(read_hist)
        ##accuracy
        train_acc = [float(i) for i in next(read_hist)]
        valid_acc = [float(i) for i in next(read_hist)]
        ##loss
        train_loss = [float(i) for i in next(read_hist)]
        valid_loss = [float(i) for i in next(read_hist)]

        plt.plot(train_acc)
        plt.plot(valid_acc)
        plt.title('model-accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        model_acc_fig_path = os.path.join(plot_path, 'model-accuracy.png')
        plt.savefig(model_acc_fig_path)
        plt.clf() ##clear the plot

        ## Plot Error
        plt.plot(train_loss)
        plt.plot(valid_loss)
        plt.title('model-loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        model_loss_fig_path = os.path.join(plot_path, 'model-loss.png')
        plt.savefig(model_loss_fig_path)


def plot_hist_data(history, prefix):
    plot_path = os.path.join(ConfigObj.Path_Parent_Dir, 'src_train_model', 'plots_logs')
    dumphist2csv(history, prefix) ##backup hist in csv file

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model-accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    model_acc_fig_path = os.path.join(plot_path, prefix + '-' + 'model-accuracy.png')
    plt.savefig(model_acc_fig_path)
    plt.clf()

    ## Plot Error
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model-loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    model_loss_fig_path = os.path.join(plot_path, prefix + '-' + 'model-loss.png')
    plt.savefig(model_loss_fig_path)

##plot_hist_frm_csv('plots_logs/hist_log.csv')