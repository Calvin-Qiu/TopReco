#!/usr/bin/env python
import tensorflow as tf
import os
import sys
import argparse
from tqdm import tqdm

import re
import time
import random
import functools
import six

import numpy as np
import sklearn.metrics


from graph_nets import utils_tf
from graph_nets import utils_np
import sonnet as snt

from types import SimpleNamespace
import tensorflow as tf
from tensorflow.compat.v1 import logging
logging.info("TF Version:{}".format(tf.__version__))


from root_gnn import model as all_models
from root_gnn.src.datasets import topreco
from root_gnn.src.datasets import graph
from root_gnn.utils import load_yaml


target_scales = np.array([65, 65, 570, 400, 1, 1]*topreco.n_max_tops).reshape((topreco.n_max_tops, -1)).T.reshape((-1,))
target_mean = np.array([0, 0, 0, 400, 0, 0]*topreco.n_max_tops).reshape((topreco.n_max_tops, -1)).T.reshape((-1,))

def read_dataset(filenames):
    """
    Read dataset...
    """
    AUTO = tf.data.experimental.AUTOTUNE
    tr_filenames = tf.io.gfile.glob(filenames)
    n_files = len(tr_filenames)

    dataset = tf.data.TFRecordDataset(tr_filenames)
    dataset = dataset.map(graph.parse_tfrec_function, num_parallel_calls=AUTO)
    return dataset, n_files


def loop_dataset(datasets, batch_size):
    if batch_size > 0:
        in_list = []
        target_list = []
        for dataset in datasets:
            inputs_tr, targets_tr = dataset
            in_list.append(inputs_tr)
            target_list.append(targets_tr)
            if len(in_list) == batch_size:
                inputs_tr = utils_tf.concat(in_list, axis=0)
                targets_tr = utils_tf.concat(target_list, axis=0)
                yield (inputs_tr, targets_tr)
    else:
        for dataset in datasets:
            yield dataset

def get_input_signature(dataset, batch_size):
    with_batch_dim = False
    input_list = []
    target_list = []
    for dd in dataset.take(batch_size).as_numpy_iterator():
        input_list.append(dd[0])
        target_list.append(dd[1])

    inputs = utils_tf.concat(input_list, axis=0)
    targets = utils_tf.concat(target_list, axis=0)
    input_signature = (
        graph.specs_from_graphs_tuple(inputs, with_batch_dim),
        graph.specs_from_graphs_tuple(targets, with_batch_dim),
        tf.TensorSpec(shape=[], dtype=tf.bool)
    )
    return input_signature

def loop_dataset(datasets, batch_size):
    if batch_size > 0:
        in_list = []
        target_list = []
        for dataset in datasets:
            inputs_tr, targets_tr = dataset
            in_list.append(inputs_tr)
            target_list.append(targets_tr)
            if len(in_list) == batch_size:
                inputs_tr = utils_tf.concat(in_list, axis=0)
                targets_tr = utils_tf.concat(target_list, axis=0)
                yield (inputs_tr, targets_tr)
                in_list = []
                target_list = []
    else:
        for dataset in datasets:
            yield dataset

def train_and_evaluate(args):
    device = 'CPU'
    gpus = tf.config.experimental.list_physical_devices("GPU")
    n_gpus = len(gpus)
    logging.info("Found {} GPUs".format(n_gpus))
    distributed = args.distributed

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if len(gpus) > 0:
        device = "{} GPUs".format(len(gpus))

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logging.info("Checkpoints and models saved at {}".format(output_dir))

    num_processing_steps_tr = args.num_iters     ## level of message-passing
    n_epochs = args.max_epochs
    logging.info("{} epochs with batch size {}".format(n_epochs, args.batch_size))
    logging.info("{} processing steps in the model".format(num_processing_steps_tr))

    train_input_dir = os.path.join(args.input_dir, 'train') 
    val_input_dir = os.path.join(args.input_dir, 'val')
    train_files = tf.io.gfile.glob(os.path.join(train_input_dir, args.patterns))
    eval_files = tf.io.gfile.glob(os.path.join(val_input_dir, args.patterns))

    if distributed:
        logging.info("Doing distributed training on {} GPUS".format(n_gpus))
        strategy = tf.distribute.MirroredStrategy()

    AUTO = tf.data.experimental.AUTOTUNE
    training_dataset, ngraphs_train = read_dataset(train_files)
    training_dataset = training_dataset.prefetch(AUTO)
    input_signature = get_input_signature(training_dataset, args.batch_size)
    learning_rate = args.learning_rate

    if distributed:
        with strategy.scope():
            optimizer = tf.keras.optimizers.Adam(learning_rate)
            model = getattr(all_models, 'FourTopPredictor')()
            checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
            ckpt_manager = tf.train.CheckpointManager(checkpoint, directory=output_dir,
                                        max_to_keep=5, keep_checkpoint_every_n_hours=8)
            logging.info("Loading latest checkpoint from: {}".format(output_dir))
            _ = checkpoint.restore(ckpt_manager.latest_checkpoint)
    else:
        learning_rate = args.learning_rate
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        model = getattr(all_models, 'FourTopPredictor')()
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        ckpt_manager = tf.train.CheckpointManager(checkpoint, directory=output_dir,
                                    max_to_keep=5, keep_checkpoint_every_n_hours=8)
        logging.info("Loading latest checkpoint from: {}".format(output_dir))
        _ = checkpoint.restore(ckpt_manager.latest_checkpoint)

    # training loss
    def loss_fcn(target_op, output_ops):
        # print("target size: ", target_op.nodes.shape)
        # print("output size: ", output_ops[0].nodes.shape)
        # output_op = output_ops[-1]
        # print("loss of 4-vect: ", tf.nn.l2_loss((target_op.nodes[:, :4] - output_op.nodes[:topreco.n_max_tops, :4])))
        # print("loss of charge: ", tf.math.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(tf.cast(target_op.nodes[:, 4:6], tf.int32),  output_op.nodes[:topreco.n_max_tops, 4:6])))
        # print("loss of predictions: ", tf.compat.v1.losses.log_loss(tf.cast(target_op.nodes[:, 6], tf.int32),  tf.math.sigmoid(output_op.nodes[:topreco.n_max_tops, 6])))

        # loss_ops = [tf.nn.l2_loss((target_op.nodes[:, :4] - output_op.nodes[:topreco.n_max_tops, :4]))
        #     + tf.math.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(tf.cast(target_op.nodes[:, 4:6], tf.int32),  output_op.nodes[:topreco.n_max_tops, 4:6]))
        #     + tf.compat.v1.losses.log_loss(tf.cast(target_op.nodes[:, 6], tf.int32),  tf.math.sigmoid(output_op.nodes[:topreco.n_max_tops, 6]))
        #     for output_op in output_ops
        # ]

        # loss_ops = [tf.nn.l2_loss((target_op.globals[:, :topreco.n_max_tops*4] - output_op.globals[:, :topreco.n_max_tops*4])) / target_op.globals.shape[0]
        #     + tf.compat.v1.losses.log_loss(
        #         tf.cast(target_op.globals[:, topreco.n_max_tops*4:topreco.n_max_tops*5], tf.int32),\
        #         tf.math.sigmoid(output_op.globals[:, topreco.n_max_tops*4:topreco.n_max_tops*5]))
        #     + tf.compat.v1.losses.log_loss(
        #         tf.cast(target_op.globals[:, topreco.n_max_tops*5:], tf.int32),\
        #         tf.math.sigmoid(output_op.globals[:, topreco.n_max_tops*5:]))
        #     for output_op in output_ops
        # ]
        
        ###### RGN loss #####
        # alpha / 4 * |dP|^2 = |dP|^2 / (2 * sigma^2)
        # ==> alpha  = 2 / sigma^2 = 200 if sigma = 0.1
        
        mask = target_op.globals[:, topreco.n_max_tops*(topreco.n_target_node_features - 1):] # indicator for real top
        n_target_top = tf.reduce_sum(mask, 1)
        alpha = tf.constant(1, dtype=tf.float32)
        batch_size = target_op.globals.shape[0]
        tops_true = tf.reshape(target_op.globals, [batch_size, topreco.n_target_node_features, topreco.n_max_tops])
        tops_true = tf.einsum('ijk->ikj', tops_true)

        tops_preds = output_ops

        loss_ops = []
        eps = 1e-5
        for i in range(topreco.n_max_tops):
            top_preds = tops_preds[i]
            for top_pred in top_preds:
                four_vec_pred = top_pred.globals[:, :4]
                four_vec_true = tops_true[:, i, :4]
                diff = four_vec_true - four_vec_pred
                target_exist = tf.cast((i < n_target_top), dtype=tf.float32)
                diff = tf.einsum('ij,i->ij', diff, target_exist)
                # loss = tf.math.reduce_mean(tf.math.abs(diff)) \
                # - tf.compat.v1.log(eps + tf.math.sigmoid(top_pred.globals[:, -1]))
                loss = tf.compat.v1.losses.mean_squared_error(diff, tf.zeros_like(diff))
                loss_ops.append(alpha * loss)
                # end of sequence loss
                # loss_eos = - tf.compat.v1.log(eps + 1 - tf.math.sigmoid(top_pred.globals[:, -1]))
                # loss_ops.append(loss_eos * tf.cast((i == n_target_top), dtype=tf.float32)) # FIX ME: this is an outer-product
        
        ###### L1 4-vector regression loss #####
        # loss_ops = [ tf.compat.v1.losses.absolute_difference(
        #                     target_op.globals[:, :4],\
        #                     output_op.globals[:, :4])
        #     for output_op in output_ops[0] # taking first top only
        # ]

        total_loss = tf.stack(loss_ops)
        return tf.math.reduce_sum(total_loss) / tf.constant(num_processing_steps_tr, dtype=tf.float32) / topreco.n_max_tops

    # @functools.partial(tf.function, input_signature=input_signature)
    def train_step(data):
        inputs_tr, targets_tr = data
        with tf.GradientTape() as tape:
            outputs_tr = model(inputs_tr, num_processing_steps_tr, is_training=True)
            loss_op_tr = loss_fcn(targets_tr, outputs_tr)
            if distributed:
                loss_op_tr /= strategy.num_replicas_in_sync # so that we don't multiply loss by number of workers after reducing

        gradients = tape.gradient(loss_op_tr, model.trainable_variables)
        # optimizer.apply(gradients, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return loss_op_tr

    @functools.partial(tf.function, input_signature=input_signature)
    def distributed_train_step(distributed_data):
      per_replica_losses = strategy.run(train_step, args=(distributed_data))
      return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


    def train_epoch(dataset):
        total_loss = 0.
        batch = 0

        # for inputs in tqdm(loop_dataset(dataset, args.batch_size)):
        #     inputs_tr, targets_tr = inputs
        #     new_target = (targets_tr.globals - target_mean) / target_scales
        #     targets_tr = targets_tr.replace(globals=new_target)
        #     if distributed:
        #         def dist_data_fn(ctx):
        #             num_replicas = ctx.num_replicas_in_sync
        #             replica_id = tf.distribute.get_replica_context().replica_id_in_sync_group
        #             # batch_size_per_replica = args.batch_size // num_replicas # if there is non-zero remainder, leftover data will be ignored
        #             replica_batch_size = ctx.get_per_replica_batch_size(args.batch_size)
        #             dist_inputs = utils_tf.get_graph(inputs_tr, slice(replica_batch_size * replica_id, replica_batch_size * (replica_id + 1)))
        #             dist_targets = utils_tf.get_graph(targets_tr, slice(replica_batch_size * replica_id, replica_batch_size * (replica_id + 1)))
        #             return (dist_inputs, dist_targets)
        #         distributed_data = strategy.experimental_distribute_datasets_from_function(dist_data_fn)
        #         total_loss += distributed_train_step(distributed_data).numpy()
        #     else:
        #         total_loss += train_step((inputs_tr, targets_tr)).numpy()
        #     batch += 1

        dataset = dataset.batch(batch_size=args.batch_size, drop_remainder=True)
        dist_dataset = strategy.experimental_distribute_dataset(dataset)
        for data in dist_dataset:
            total_loss += distributed_train_step(dist_dataset).numpy()
            batch += 1

        logging.info("total batches: {}".format(batch))
        return total_loss/batch, batch
        # return total_loss/batch/args.batch_size, batch

    out_str  = "Start training " + time.strftime('%d %b %Y %H:%M:%S', time.localtime())
    out_str += '\n'
    out_str += f"lr = {learning_rate}\n"
    out_str += "Epoch, Time [mins], Loss\n"
    log_name = os.path.join(output_dir, "training_log.txt")
    with open(log_name, 'a') as f:
        f.write(out_str)
    now = time.time()

    for epoch in range(n_epochs):
        logging.info("start epoch {} on {}".format(epoch, device))

        # shuffle the dataset before training
        training_dataset = training_dataset.shuffle(args.shuffle_size, seed=12345, reshuffle_each_iteration=True)
        loss,batches = train_epoch(training_dataset)
        this_epoch = time.time()

        logging.info("{} epoch takes {:.2f} mins with loss {:.4f} in {} batches".format(
            epoch, (this_epoch-now)/60., loss, batches))
        out_str = "{}, {:.2f}, {:.4f}\n".format(epoch, (this_epoch-now)/60., loss)
        now = this_epoch
        with open(log_name, 'a') as f:
            f.write(out_str)
        ckpt_manager.save()

    out_log = "End @ " + time.strftime('%d %b %Y %H:%M:%S', time.localtime()) + "\n"
    with open(log_name, 'a') as f:
        f.write(out_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train nx-graph with configurations')
    add_arg = parser.add_argument
    add_arg("--input-dir", help='input directory that contains subfolder of train, val and test')
    add_arg("--patterns", help='file patterns', default='*')
    add_arg("--output-dir", help="where the model and training info saved")
    add_arg('-d', '--distributed', action='store_true', help='data distributed training')
    add_arg("--num-iters", help="number of message passing steps", default=8, type=int)
    add_arg("--learning-rate", help='learing rate', default=0.0005, type=float)
    add_arg("--max-epochs", help='number of epochs', default=1, type=int)
    add_arg("--batch-size", type=int, help='training/evaluation batch size', default=500)
    add_arg("--shuffle-size", type=int, help="number of events for shuffling", default=650)
    add_arg("-v", "--verbose", help='verbosity', choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],\
        default="INFO")
    args, _ = parser.parse_known_args()

    # Set python level verbosity
    logging.set_verbosity(args.verbose)
    # Suppress C++ level warnings.
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    train_and_evaluate(args)