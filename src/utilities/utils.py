import os
import yaml
import pickle

import tensorflow as tf
import seaborn as sns
import numpy as np
import operator
import matplotlib.pyplot as plt


def one_hot_to_mask(one_hot):
    n_classes = one_hot.shape[-1]
    idx = tf.argmax(one_hot, axis=-1)
    colors = sns.color_palette('hls', n_classes)
    multi_mask = tf.gather(colors, idx)
    multi_mask = np.where(multi_mask[:,:,:]==colors[0], 0, multi_mask[:,:,:])
   
    return multi_mask


def resize_image(dim, factor=2048, threshold=0, op=operator.gt):
    boundaries = [factor*i for i in range(100000)]
    boundaries = [f for f in boundaries if op(f,threshold)]
    diff = list(map(lambda x: abs(dim-x), boundaries))
    new_dim = boundaries[diff.index(min(diff))]

    return new_dim


def get_train_curves(history,train_metric,valid_metric,save_path):
    sns.set_style('dark')
    fig = plt.figure(figsize=(8,5))
    epochs = range(len(history[train_metric]))

    #plot train
    sns.lineplot(epochs,
                 history[train_metric],
                 markers=True,
                 dashes=False,
                 label='Training'+train_metric)
    
    #plot validation
    sns.lineplot(range(len(history[valid_metric])),
                 history[valid_metric],
                 markers=True,
                 dashes=False,
                 label='Validation'+valid_metric)

    plt.title('Training and validation'+train_metric)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.grid()
    fig.savefig(os.path.join(save_path,train_metric+'.png'))
    plt.close()


def get_files(files_path, ext):
    files_lst=[]
    for path, subdirs, files in os.walk(files_path):
        for name in files:
            if name.endswith(ext):
                files_lst.append(os.path.join(path,name))
    return files_lst


def save_experiment(model,config,history,model_save_path):
    model.save(os.path.join(model_save_path,'model_best.h5'))

    with open(os.path.join(model_save_path,'history'), 'wb') as history_file:
        pickle.dump(history, history_file)

    with open(os.path.join(model_save_path,'config.yaml'), 'w') as config_file:
        yaml.dump(config, config_file)





