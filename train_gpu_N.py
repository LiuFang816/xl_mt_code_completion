from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import time
import reader_pointer_mt2 as reader
from absl import flags


import tensorflow as tf
import model_N as model

from gpu_utils import assign_to_gpu, average_grads_and_vars

import numpy as np

N_filename = '/data/liufang/mt/PY_data/pickle_data/PY_non_terminal.pickle'
T_filename = '/data/liufang/mt/PY_data/pickle_data/PY_terminal_50k_whole.pickle'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# GPU config
flags.DEFINE_integer("num_hosts", default=1,
                     help="Number of TPU hosts")
flags.DEFINE_integer("num_core_per_host", default=3,
                     help="Number of cores per host")

flags.DEFINE_string("model_dir", default=None,
                    help="Estimator model_dir.")
flags.DEFINE_bool("do_train", default=True,
                  help="Whether to run training.")
flags.DEFINE_bool("do_eval", default=False,
                  help="Whether to run eval on the dev set.")
flags.DEFINE_string("eval_ckpt_path", None,
                    help="Checkpoint path for do_test evaluation."
                         "If set, model_dir will be ignored."
                         "If unset, will use the latest ckpt in model_dir.")
flags.DEFINE_string("warm_start_path", None,
                    help="Checkpoint path for warm start."
                         "If set, will clear Adam states."
                         "Note that the new model_dir should be different"
                         " from warm_start_path.")
flags.DEFINE_string("gpu", "1", "gpu id")

# Optimization config
flags.DEFINE_float("learning_rate", default=2.5e-4,
                   help="Maximum learning rate.")
flags.DEFINE_float("clip", default=0.25,
                   help="Gradient clipping value.")
# for cosine decay
flags.DEFINE_float("min_lr_ratio", default=0.004,
                   help="Minimum ratio learning rate.")
flags.DEFINE_integer("warmup_steps", default=0,
                     help="Number of steps for linear lr warmup.")

# Training config
flags.DEFINE_integer("epochs", default=30,
                     help="epoch size.")
flags.DEFINE_integer("train_batch_size", default=120,
                     help="Size of train batch.")
flags.DEFINE_integer("eval_batch_size", default=60,
                     help="Size of valid batch.")
flags.DEFINE_integer("train_steps", default=100000,
                     help="Total number of training steps.")
flags.DEFINE_integer("iterations", default=500,
                     help="Number of iterations per repeat loop.")
flags.DEFINE_integer("save_steps", default=10000,
                     help="number of steps for model checkpointing.")

# Evaluation config
flags.DEFINE_bool("do_test", default=False,
                  help="Run on the test set.")
flags.DEFINE_integer("max_eval_batch", default=-1,
                     help="Set -1 to turn off. Only used in test mode.")
flags.DEFINE_bool("do_eval_only", default=False,
                  help="Run evaluation only.")
flags.DEFINE_integer("start_eval_steps", default=10000,
                     help="Which checkpoint to start with in `do_eval_only` mode.")
flags.DEFINE_string("eval_split", "valid",
                    help="Which data split to evaluate.")

# Model config
flags.DEFINE_integer("tgt_len", default=256,
                     help="Number of steps to predict")
flags.DEFINE_integer("mem_len", default=256,
                     help="Number of steps to cache")
flags.DEFINE_bool("same_length", default=False,
                  help="Same length attention")
flags.DEFINE_integer("clamp_len", default=-1,
                     help="Clamp length")

flags.DEFINE_integer("n_layer", default=6,
                     help="Number of layers.")
flags.DEFINE_integer("d_model_N", default=300,
                     help="Dimension of the model.")
flags.DEFINE_integer("d_embed_N", default=300,
                     help="Dimension of the embeddings.")
flags.DEFINE_integer("d_model_T", default=1200,
                     help="Dimension of the model.")
flags.DEFINE_integer("d_embed_T", default=1200,
                     help="Dimension of the embeddings.")
flags.DEFINE_integer("n_head", default=5,
                     help="Number of attention heads.")
flags.DEFINE_integer("d_head", default=64,
                     help="Dimension of each attention head.")
flags.DEFINE_integer("d_inner", default=1024,
                     help="Dimension of inner hidden size in positionwise feed-forward.")
flags.DEFINE_float("dropout", default=0.1,
                   help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.1,
                   help="Attention dropout rate.")
flags.DEFINE_bool("untie_r", default=False,
                  help="untie r_w_bias and r_r_bias")
flags.DEFINE_float("alpha", default=0.7,
                   help="Type loss weight")

# Adaptive Softmax / Embedding
flags.DEFINE_bool("tie_weight", default=True,
                  help="Tie embedding and softmax weight.")
flags.DEFINE_integer("div_val", default=1,
                     help="Divide the embedding size by this val for each bin")
flags.DEFINE_bool("proj_share_all_but_first", default=False,
                  help="True to share all but first projs, False not to share.")
flags.DEFINE_bool("proj_same_dim", default=True,
                  help="Project the bin with the same dimension.")

# Parameter initialization
flags.DEFINE_enum("init", default="normal",
                  enum_values=["normal", "uniform"],
                  help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02,
                   help="Initialization std when init is normal.")
flags.DEFINE_float("proj_init_std", default=0.01,
                   help="Initialization std for embedding projection.")
flags.DEFINE_float("init_range", default=0.1,
                   help="Initialization std when init is uniform.")

FLAGS = flags.FLAGS

gpu_list = [0, 1, 2]


def get_model_fn(n_token_N, n_token_T, cutoffs, alpha):
    def model_fn(inpN, inpT, tgtN, mems, is_training):
        inpN = tf.transpose(inpN, [1, 0])
        inpT = tf.transpose(inpT, [1, 0])
        tgtN = tf.transpose(tgtN, [1, 0])

        if FLAGS.init == "uniform":
            initializer = tf.initializers.random_uniform(
                minval=-FLAGS.init_range,
                maxval=FLAGS.init_range,
                seed=None)
        elif FLAGS.init == "normal":
            initializer = tf.initializers.random_normal(
                stddev=FLAGS.init_std,
                seed=None)
            proj_initializer = tf.initializers.random_normal(
                stddev=FLAGS.proj_init_std,
                seed=None)

        tie_projs = [False for _ in range(len(cutoffs) + 1)]
        if FLAGS.proj_share_all_but_first:
            for i in range(1, len(tie_projs)):
                tie_projs[i] = True

        lossN, new_mems, predictionN = model.transformer(
            inpN=inpN,
            inpT=inpT,
            targetsN=tgtN,
            mems=mems,
            n_token_N=n_token_N,
            n_token_T=n_token_T,
            n_layer=FLAGS.n_layer,
            d_model_N=FLAGS.d_model_N,
            d_model_T=FLAGS.d_model_T,
            d_embed_N=FLAGS.d_embed_N,
            d_embed_T=FLAGS.d_embed_T,
            n_head=FLAGS.n_head,
            d_head=FLAGS.d_head,
            d_inner=FLAGS.d_inner,
            dropout=FLAGS.dropout,
            dropatt=FLAGS.dropatt,
            initializer=initializer,
            proj_initializer=proj_initializer,
            is_training=is_training,
            mem_len=FLAGS.mem_len,
            cutoffs=cutoffs,
            div_val=FLAGS.div_val,
            tie_projs=tie_projs,
            input_perms=None,
            target_perms=None,
            head_target=None,
            same_length=FLAGS.same_length,
            clamp_len=FLAGS.clamp_len,
            use_tpu=False,
            untie_r=FLAGS.untie_r,
            proj_same_dim=FLAGS.proj_same_dim)

        # number of parameters
        num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
        tf.logging.info('#params: {}'.format(num_params))

        loss = lossN
        # format_str = '{{:<{0}s}}\t{{}}'.format(
        #     max([len(v.name) for v in tf.trainable_variables()]))
        # for v in tf.trainable_variables():
        #   tf.logging.info(format_str.format(v.name, v.get_shape()))

        if is_training:
            all_vars = tf.trainable_variables()
            grads = tf.gradients(loss, all_vars)
            grads_and_vars = list(zip(grads, all_vars))

            return loss, new_mems, grads_and_vars, predictionN
        else:
            return loss, new_mems, predictionN

    return model_fn


def single_core_graph(n_token_N, n_token_T, cutoffs, is_training, inpN, inpT, tgtN, mems, alpha):
    model_fn = get_model_fn(
        n_token_N=n_token_N,
        n_token_T=n_token_T,
        cutoffs=cutoffs,
        alpha=alpha)

    model_ret = model_fn(
        inpN=inpN,
        inpT=inpT,
        tgtN=tgtN,
        mems=mems,
        is_training=is_training)

    return model_ret


def train(train_data, valid_data, n_token_N, n_token_T, cutoffs, vocab_size, fout):
    ##### Get input function and model function

    input_dataN = tf.placeholder(dtype=tf.int32, shape=[FLAGS.train_batch_size, FLAGS.tgt_len], name="input_dataN")
    input_dataT = tf.placeholder(dtype=tf.int32, shape=[FLAGS.train_batch_size, FLAGS.tgt_len], name="input_dataT")

    targetsN = tf.placeholder(dtype=tf.int32, shape=[FLAGS.train_batch_size, FLAGS.tgt_len], name="targetsN")

    inputsN = tf.split(input_dataN, len(gpu_list), 0)
    inputsT = tf.split(input_dataT, len(gpu_list), 0)
    labelsN = tf.split(targetsN, len(gpu_list), 0)

    per_core_bsz = FLAGS.train_batch_size // len(gpu_list)

    tower_mems, tower_losses, tower_new_mems, tower_grads_and_vars, predictionN = [], [], [], [], []

    for i, gpu_id in enumerate(gpu_list):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=(i > 0)):
                # for i in range(FLAGS.num_core_per_host):
                #     reuse = True if i > 0 else None
                #     with tf.device(assign_to_gpu(i, ps_device)), \
                #          tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
                mems_i = [tf.placeholder(tf.float32,
                                         [FLAGS.mem_len, per_core_bsz, FLAGS.d_model_T + FLAGS.d_model_N])
                          for _ in range(FLAGS.n_layer)]

                loss_i, new_mems_i, grads_and_vars_i, predictionN_i = single_core_graph(
                    n_token_N=n_token_N,
                    n_token_T=n_token_T,
                    cutoffs=cutoffs,
                    is_training=True,
                    inpN=inputsN[i],
                    inpT=inputsT[i],
                    tgtN=labelsN[i],
                    mems=mems_i,
                    alpha=FLAGS.alpha)

                tower_mems.append(mems_i)
                tower_losses.append(loss_i)
                tower_new_mems.append(new_mems_i)
                tower_grads_and_vars.append(grads_and_vars_i)
                predictionN.append(predictionN_i)

    ## average losses and gradients across towers
    if len(tower_losses) >= 1:
        loss = tf.add_n(tower_losses) / len(tower_losses)
        predictionN = tf.concat(predictionN, axis=0)
        acc_N = tf.reduce_mean(tf.cast(predictionN, tf.float32))
        grads_and_vars = average_grads_and_vars(tower_grads_and_vars)
    else:
        loss = tower_losses[0]
        grads_and_vars = tower_grads_and_vars[0]
    grads, all_vars = zip(*grads_and_vars)

    ## clip gradient
    clipped, gnorm = tf.clip_by_global_norm(grads, FLAGS.clip)
    grads_and_vars = list(zip(clipped, all_vars))

    ## configure the optimizer
    global_step = tf.train.get_or_create_global_step()

    # warmup stage: increase the learning rate linearly
    if FLAGS.warmup_steps > 0:
        warmup_lr = tf.to_float(global_step) / tf.to_float(FLAGS.warmup_steps) \
                    * FLAGS.learning_rate
    else:
        warmup_lr = 0.0

    # decay stage: decay the learning rate using the cosine schedule
    decay_lr = tf.train.cosine_decay(
        FLAGS.learning_rate,
        global_step=global_step - FLAGS.warmup_steps,
        decay_steps=FLAGS.train_steps - FLAGS.warmup_steps,
        alpha=FLAGS.min_lr_ratio)

    # choose warmup or decay
    learning_rate = tf.where(global_step < FLAGS.warmup_steps,
                             warmup_lr, decay_lr)

    # get the train op
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step)

    ##### Training loop
    tower_mems_np = [
        [np.zeros([FLAGS.mem_len, per_core_bsz, FLAGS.d_model_N + FLAGS.d_model_T], dtype=np.float32)
         for layer in range(FLAGS.n_layer)]
        for core in range(len(gpu_list))
    ]

    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(FLAGS.epochs):
            start_time = time.time()
            min_loss = float("INF")
            if FLAGS.warm_start_path is not None:
                tf.logging.info("warm start from {}".format(FLAGS.warm_start_path))
                saver.restore(sess, FLAGS.warm_start_path)

            fetches = [loss, tower_new_mems, global_step, gnorm, learning_rate, train_op, acc_N]

            total_loss, prev_step = 0., -1
            data_loader = reader.real_data_producer(train_data, FLAGS.train_batch_size, FLAGS.tgt_len, vocab_size)
            step = 0
            while True:
                feed_dict = {}
                dataN, tN, dataT, tT, epoch_size, eof_indicator, input_dataP = next(
                    data_loader)

                feed_dict[input_dataN] = dataN
                feed_dict[input_dataT] = dataT
                feed_dict[targetsN] = tN

                for i in range(len(gpu_list)):
                    for m, m_np in zip(tower_mems[i], tower_mems_np[i]):
                        feed_dict[m] = m_np

                fetched = sess.run(fetches, feed_dict=feed_dict)

                loss_np, tower_mems_np, curr_step = fetched[:3]
                accuracy_N = (fetched[-1])
                total_loss += loss_np

                if curr_step > 0 and curr_step % FLAGS.iterations == 0:
                    curr_loss = total_loss / (curr_step - prev_step)

                    tf.logging.info("Epoch {}  |  [{}] | gnorm {:.2f} lr {:8.6f} "
                                    "| loss {:.2f} | pplx {:>7.2f}, accN {:.4f} bpc {:>7.4f}".format(
                        epoch + 1, curr_step, fetched[-4], fetched[-3],
                        curr_loss, math.exp(curr_loss), accuracy_N, curr_loss / math.log(2)))
                    total_loss, prev_step = 0., curr_step
                    print("Epoch {}  |  [{}] | gnorm {:.2f} lr {:8.6f} "
                          "| loss {:.2f} | pplx {:>7.2f}, accN {:.4f} bpc {:>7.4f}".format(
                        epoch + 1, curr_step, fetched[-4], fetched[-3],
                        curr_loss, math.exp(curr_loss), accuracy_N, curr_loss / math.log(2)), file=fout)
                    fout.flush()

                    if FLAGS.model_dir:
                        if curr_loss < min_loss:
                            min_loss = curr_loss
                            save_path = os.path.join(FLAGS.model_dir, "model.ckpt")
                            saver.save(sess, save_path)
                            tf.logging.info("Model saved in path: {}".format(save_path))

                step += 1
                if step >= epoch_size:
                    break
                    # if curr_step == FLAGS.train_steps:
                    #     break
            print('this run_epoch takes time %.2f' % (time.time() - start_time))
            # evaluate
            eval_fetches = [loss, tower_new_mems, acc_N]
            total_loss, prev_step = 0., -1
            total_accN = 0.
            data_loader = reader.real_data_producer(valid_data, FLAGS.train_batch_size, FLAGS.tgt_len, vocab_size)
            step = 0
            while True:
                feed_dict = {}
                dataN, tN, dataT, tT, epoch_size, eof_indicator, input_dataP = next(
                    data_loader)

                feed_dict[input_dataN] = dataN
                feed_dict[input_dataT] = dataT
                feed_dict[targetsN] = tN

                for i in range(len(gpu_list)):
                    for m, m_np in zip(tower_mems[i], tower_mems_np[i]):
                        feed_dict[m] = m_np

                eval_fetched = sess.run(eval_fetches, feed_dict=feed_dict)

                loss_np, tower_mems_np = eval_fetched[:2]
                accuracy_N = (eval_fetched[-1])
                total_accN += accuracy_N
                total_loss += loss_np

                step += 1
                if step >= epoch_size:
                    break

            tf.logging.info(
                "Validation:  Epoch {}  | loss {:.2f} | pplx {:>7.2f}, accN {:.4f}, bpc {:>7.4f}".format(
                    epoch + 1, total_loss / epoch_size, math.exp(total_loss / epoch_size), total_accN / epoch_size,
                    (total_loss / epoch_size) / math.log(2)))
            print("Validation:  Epoch {}  | loss {:.2f} | pplx {:>7.2f}, accN {:.4f}, bpc {:>7.4f}".format(
                epoch + 1, total_loss / epoch_size, math.exp(total_loss / epoch_size), total_accN / epoch_size,
                (total_loss / epoch_size) / math.log(2)), file=fout)
            fout.flush()


def main(unused_argv):
    outfile = 'logs/PY_output_N_{}.txt'.format(FLAGS.alpha)
    fout = open(outfile, 'a')

    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    del unused_argv  # Unused

    train_dataN, valid_dataN, vocab_sizeN, train_dataT, valid_dataT, vocab_sizeT, attn_size, train_dataP, valid_dataP = reader.input_data(
        N_filename, T_filename)

    train_data = (train_dataN, train_dataT, train_dataP)
    valid_data = (valid_dataN, valid_dataT, valid_dataP)
    vocab_size = (vocab_sizeN + 1, vocab_sizeT + 2)  # plus EOF, N is [w, eof], T is [w, unk, eof]
    n_token_N, n_token_T = vocab_size
    tf.logging.info("n_token N {}".format(n_token_N))
    tf.logging.info("n_token T {}".format(n_token_T))

    tf.logging.set_verbosity(tf.logging.INFO)

    train_dataN, train_dataP, train_dataT_x, train_dataT_y = reader.raw_data_producer(train_data,
                                                                                      FLAGS.train_batch_size,
                                                                                      FLAGS.tgt_len,
                                                                                      vocab_size,
                                                                                      change_yT=True)

    train_data = (train_dataN, train_dataP, train_dataT_x, train_dataT_y)
    valid_dataN, valid_dataP, valid_dataT_x, valid_dataT_y = reader.raw_data_producer(valid_data,
                                                                                      FLAGS.train_batch_size,
                                                                                      FLAGS.tgt_len,
                                                                                      vocab_size,
                                                                                      change_yT=True)

    valid_data = (valid_dataN, valid_dataP, valid_dataT_x, valid_dataT_y)


    if FLAGS.do_train:
        train(train_data, valid_data, n_token_N, n_token_T, [], vocab_size, fout)



if __name__ == "__main__":
    tf.app.run()
