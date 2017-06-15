import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from mpcnn import MPCNN
from tensorflow.contrib import learn
from tensorflow.python import debug as tf_debug


# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("training_datafilename", "data/msrparaphrase/msr_paraphrase_train_processed.txt", "training data file containing sentence pairs and similarity score in one line")
#NOTE: Example of a line in data_file(the three arguments are '$$' seperated)
#a masterpiece four years in the making $$ It took this masterpiece four years to make $$ 1 


# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3", "Comma-separated filter sizes (default: '1,2,3')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size ")
tf.flags.DEFINE_integer("num_epochs", 30, "Number of training epochs ")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 30, "Number of checkpoints to store ")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("debug", False, "Run with tf debugger")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


# Data Preparation
# ==================================================

# Load data
print("Loading data...")


x1_text,x2_text, y = data_helpers.load_data_and_labels(FLAGS.training_datafilename)
x_combined = x1_text + x2_text
# Build vocabulary
max_document_length = max([len(x1.split(" ")) for x1 in x_combined])
print("max_document_length: ",max_document_length)
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

x_combined_indexes = list(vocab_processor.fit_transform(x_combined))
x1 = np.array(x_combined_indexes[:len(x_combined_indexes)/2])
x2 = np.array(x_combined_indexes[len(x_combined_indexes)/2:])

# Split train/test set
dev_sample_index = -1 * (int(FLAGS.dev_sample_percentage * float(len(y)))) 
x1_train, x1_dev = x1[:dev_sample_index], x1[dev_sample_index:]
x2_train, x2_dev = x2[:dev_sample_index], x2[dev_sample_index:]
y_train, y_dev = y[:dev_sample_index], y[dev_sample_index:]
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


#Add max sentence length as a filter size
FLAGS.filter_sizes+=",{}".format(str(max_document_length))


print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement
    )
    session_conf.gpu_options.allow_growth = True


    sess = tf.Session(config=session_conf)
    with sess.as_default():
        if FLAGS.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        mpcnn = MPCNN(
            inp1_sequence_length=x1_train.shape[1],
            inp2_sequence_length=x2_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters_A=FLAGS.num_filters,
            num_filters_B=FLAGS.num_filters
        )


        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(mpcnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                pass
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                var_hist_summary = tf.summary.histogram("{}/var/hist".format(v.name), v)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(var_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)
        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", mpcnn.loss)
        acc_summary = tf.summary.scalar("accuracy", mpcnn.accuracy)

        # Train Summaries
        #train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        check = tf.add_check_numerics_ops()

        def train_step(x1_batch,x2_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              mpcnn.input_x1: x1_batch,
              mpcnn.input_x2: x2_batch,
              mpcnn.input_y: y_batch
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, mpcnn.loss, mpcnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x1_batch, x2_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              mpcnn.input_x1: x1_batch,
              mpcnn.input_x2: x2_batch,
              mpcnn.input_y: y_batch
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, mpcnn.loss, mpcnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            if writer:
                writer.add_summary(summaries, step)

            return loss, accuracy

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x1_train,x2_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x1_batch,x2_batch, y_batch = zip(*batch)
            train_step(x1_batch, x2_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                total_dev_loss = 0.0
                total_dev_accuracy = 0.0
                
                print("\nEvaluation:")
                dev_batches = data_helpers.batch_iter(list(zip(x1_dev, x2_dev, y_dev)), FLAGS.batch_size, 1)
                for dev_batch in dev_batches:
                    x1_dev_batch, x2_dev_batch, y_dev_batch = zip (*dev_batch)
                    dev_loss,dev_accuracy = dev_step(x1_dev_batch, x2_dev_batch, y_dev_batch, writer=dev_summary_writer)
                    total_dev_loss+= dev_loss
                    total_dev_accuracy += dev_accuracy
                total_dev_accuracy = total_dev_accuracy/(len(y_dev)/FLAGS.batch_size)
                print("dev_loss {:g}, dev_acc {:g}, num_dev_batches {:g}".format( total_dev_loss, total_dev_accuracy, len(y_dev)/FLAGS.batch_size))
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
