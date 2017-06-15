from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np


from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


seed = 128
rng = np.random.RandomState(seed)
hidden_num_units = 512
output_num_units = 2

class Reduction(object):
  """Types of loss reduction."""
  # Un-reduced weighted losses with the same shape as input.
  NONE = "none"
  # Scalar sum of `NONE`.
  SUM = "weighted_sum"
  # Scalar `SUM` divided by sum of weights.
  MEAN = "weighted_mean"
  # Scalar `SUM` divided by number of non-zero weights.
  SUM_BY_NONZERO_WEIGHTS = "weighted_sum_by_nonzero_weights"
  @classmethod
  def all(cls):
    return (
        cls.NONE,
        cls.SUM,
        cls.MEAN,
        cls.SUM_BY_NONZERO_WEIGHTS)
  @classmethod
  def validate(cls, key):
    if key not in cls.all():
      raise ValueError("Invalid ReductionKey %s." % key)



def cosine_distance(
    labels, predictions, dim=None, weights=1.0, scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
  """Adds a cosine-distance loss to the training procedure.
  Note that the function assumes that `predictions` and `labels` are already
  unit-normalized.
  Args:
    labels: `Tensor` whose shape matches 'predictions'
    predictions: An arbitrary matrix.
    dim: The dimension along which the cosine distance is computed.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `losses` dimension).
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.
    reduction: Type of reduction to apply to loss.
  Returns:
    Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
    shape as `labels`; otherwise, it is scalar.
  Raises:
    ValueError: If `predictions` shape doesn't match `labels` shape, or
      `weights` is `None`.
  """
  if dim is None:
    raise ValueError("`dim` cannot be None.")
  with ops.name_scope(scope, "cosine_distance",(predictions, labels, weights)) as scope:
    predictions = math_ops.to_float(predictions)
    labels = math_ops.to_float(labels)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    radial_diffs = math_ops.multiply(predictions, labels)
    losses = 1 - math_ops.reduce_sum(radial_diffs, axis=(dim,), keep_dims=True)
    return losses

class MPCNN(object):

    def groupa (self, filter_size, embedding_size, num_filters, embedded_chars_expanded, sequence_length,reuse_flag =False):#, avgpooled_outputs, maxpooled_outputs):
        # convolution layer
        with tf.variable_scope('groupA_filters_biases',reuse = reuse_flag):
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            
            filter_avgpool = tf.get_variable("filter_avgpool",filter_shape)
            filter_minpool = tf.get_variable("filter_minpool",filter_shape)
            filter_maxpool = tf.get_variable("filter_maxpool",filter_shape)

            b_minpool = tf.get_variable("b_minpool",initializer =0.1*tf.ones([num_filters]))#,trainable = False )
            b_maxpool = tf.get_variable("b_maxpool",initializer =0.1*tf.ones([num_filters]))#,trainable = False )
            b_avgpool = tf.get_variable("b_avgpool",initializer =0.1*tf.ones([num_filters]))#,trainable = False )

        conv_avgpool = tf.nn.conv2d(
            embedded_chars_expanded,
            filter_avgpool,
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="conv_avgpool")
        h_avgpool = tf.nn.tanh(tf.nn.bias_add(conv_avgpool, b_avgpool), name="tanh_avgpool")

        avgpooled = tf.nn.avg_pool(
            h_avgpool,
            ksize=[1, sequence_length - filter_size + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="avgpool")

        conv_maxpool = tf.nn.conv2d(
            embedded_chars_expanded,
            filter_maxpool,
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="conv_maxpool")
        h_maxpool = tf.nn.tanh(tf.nn.bias_add(conv_maxpool, b_maxpool), name="tanh_maxpool")

        maxpooled = tf.nn.max_pool(
            h_maxpool,
            ksize=[1, sequence_length - filter_size + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="maxpool")

        conv_minpool = tf.nn.conv2d(
            embedded_chars_expanded,
            filter_minpool,
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="conv_minpool")
        h_minpool = tf.nn.tanh(tf.nn.bias_add(conv_minpool, b_minpool), name="tanh_minpool")


        minpooled = -tf.nn.max_pool(
            -h_minpool,
            ksize=[1, sequence_length - filter_size + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="minpool")



        avgpooled_squeezed= tf.squeeze(avgpooled, axis = [1,2]) #shape is (MB x Num_Filters)
        maxpooled_squeezed = tf.squeeze(maxpooled, axis = [1,2])
        minpooled_squeezed = tf.squeeze(minpooled, axis = [1,2])

        return avgpooled_squeezed , maxpooled_squeezed ,minpooled_squeezed

    def normalise(self, a):
        with tf.name_scope("normalise"):
            norm_of_a = tf.norm(a, axis=-1)
            norm_of_a = tf.expand_dims(norm_of_a,-1)
        return tf.divide(a, norm_of_a)


    def Algo1 (self,inp1_sentences,inp2_sentences):
        with tf.name_scope("Algo1"):
            #shape of inp1_sentence and inp2_sentences is [ types_of_poolings  x types_of_filter_sizes  x  MB xNum_Filters_for_each_size]
            feah = []
            for p in xrange(len(inp1_sentences)):  # for each pooling type p:
                inp1_sentences_stacked = tf.stack(inp1_sentences[p],name="stack_inp1_sentences") #Shape =  types_of_filter_sizes  x MB x Num_Filters_for_each_size
                inp1_sentences_stacked = tf.transpose(inp1_sentences_stacked, perm=[1,2,0]) # MB X  Num_Filters_for_each_size X types_of_filter_sizes   
                inp2_sentences_stacked = tf.stack(inp2_sentences[p],name="stack_inp2_sentences")
                inp2_sentences_stacked = tf.transpose(inp2_sentences_stacked, perm=[1,2,0])  # MB X  Num_Filters_for_each_size X types_of_filter_sizes
                inp1_sentence_normalised = self.normalise(inp1_sentences_stacked)    # MB X  Num_Filters_for_each_size X types_of_filter_sizes
                inp2_sentence_normalised = self.normalise(inp2_sentences_stacked)

                cd = tf.squeeze(cosine_distance(inp1_sentence_normalised, inp2_sentence_normalised, dim=-1,reduction=Reduction.NONE), axis=[-1])    #MB X  Num_Filters_for_each_size
                l2diff = tf.sqrt(tf.reduce_sum(tf.square(tf.clip_by_value(tf.subtract(inp1_sentences_stacked,inp2_sentences_stacked,name= "subtract_inp1_inp2"),1e-7,1e+10)),reduction_indices=-1))    ##MB X  Num_Filters_for_each_size
                feah.append(tf.concat([cd, l2diff],axis=-1))
            feah_tensor = tf.transpose(tf.stack(feah),[1,0,2])
            feah_tensor = tf.reshape(feah_tensor,[tf.shape(feah_tensor)[0],-1])
        return feah_tensor  #   shape is   MB x (p*2*filter_number)
        
    def Algo2 (self,Ga_inp1_sentences,Ga_inp2_sentences,Gb_inp1_sentences,Gb_inp2_sentences):
        with tf.name_scope("Algo2"):
            feaa = []
            feab = []
            for p in xrange(len(Ga_inp1_sentences)):  # for each pooling type p:
                for ws1 in xrange(len(Ga_inp1_sentences[p])):
                    oG1a = Ga_inp1_sentences[p][ws1] #shape is MB xNum_Filters_for_each_size
                    for ws2 in xrange(len(Ga_inp2_sentences[p])):
                        oG2a = Ga_inp2_sentences[p][ws2] #shape is MB xNum_Filters_for_each_size
                        oG1a_normalised = self.normalise(oG1a)
                        oG2a_normalised = self.normalise(oG2a)

                        displacement  = tf.sqrt(tf.reduce_sum(tf.square(tf.clip_by_value(tf.subtract(oG1a,oG2a),1e-7,1e+10)),-1,keep_dims=True))
                        cd = cosine_distance(oG1a_normalised, oG2a_normalised, dim=-1,reduction=Reduction.NONE)
                        l2diff = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(tf.clip_by_value(tf.subtract(oG1a,oG2a),1e-7,1e+10)),reduction_indices=-1)),-1)
                        feaa.append(tf.concat([cd, l2diff,displacement],axis=-1))

            for p2 in xrange(len(Gb_inp1_sentences)):  # for each pooling type p2:
                for ws1 in xrange(len(Gb_inp1_sentences[p2])):  
                    oG1b = Gb_inp1_sentences[p2][ws1] #shape is MB x embed_dim x Num_filter
                    oG2b = Gb_inp2_sentences[p2][ws1] 
                    oG1b = tf.transpose(oG1b,[0,2,1]) #shape is MB x Num_filter x embed_dim
                    oG2b = tf.transpose(oG2b,[0,2,1])
                    oG1b_normalised = self.normalise(oG1b)
                    oG2b_normalised = self.normalise(oG2b)
                    displacement2 = tf.sqrt(tf.reduce_sum(tf.square(tf.clip_by_value(tf.subtract(oG1b,oG2b),1e-7,1e+10)),-1))    #  MB x Num_filter
                    cd2 = tf.squeeze(cosine_distance(oG1b_normalised, oG2b_normalised, dim=-1,reduction=Reduction.NONE), axis=[-1])  #  MB x Num_filter
                    l2diff2 = tf.sqrt(tf.reduce_sum(tf.square(tf.clip_by_value(tf.subtract(oG1b,oG2b),1e-7,1e+10)),reduction_indices=-1))    #  MB x Num_filter
                    feab.append(tf.concat([cd2, l2diff2,displacement2],axis=-1))


            feaa_tensor = tf.stack(feaa)  
            feaa_tensor = tf.transpose(tf.stack(feaa),[1,0,2])  # MB x (p*ws1*ws2) x 3
            feaa_tensor = tf.reshape(feaa_tensor,[tf.shape(feaa_tensor)[0],-1]) # MB x (p*ws1*ws2*3)

            feab_tensor = tf.stack(feab)  #(p2*ws1)  x MB x (3*Num_filter)
            feab_tensor = tf.transpose(tf.stack(feab),[1,0,2])  #MB x (p2*ws1) x (3*Num_filter)
            feab_tensor = tf.reshape(feab_tensor,[tf.shape(feab_tensor)[0],-1])  #MB x (p2*ws1*3*Num_filter)
        
        return feaa_tensor,feab_tensor


    def __init__(self, inp1_sequence_length, inp2_sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters_A,num_filters_B):
        # Placeholders for input, output and dropout
        self.input_x1 = tf.placeholder(tf.int32, [None, inp1_sequence_length], name="input_x1")
        self.input_x2 = tf.placeholder(tf.int32, [None, inp2_sequence_length], name="input_x2")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

        with tf.device('cpu:0'):
            with tf.name_scope("input_embedding1"):
                W = tf.get_variable("W",[vocab_size,embedding_size])

                self.embedded_chars_inp1 = tf.nn.embedding_lookup(W, self.input_x1,name ="embedded_chars_inp1")
                self.embedded_chars_inp1_expanded = tf.expand_dims(self.embedded_chars_inp1, axis=-1, name="embedded_chars_inp1_expanded") #conv2d needs 4d tensor of shape [batch, width(inp1_seq_len),
                                                                                       #height(embedding_size), channel(adding artifically here)].
                
                self.embedded_chars_inp2 = tf.nn.embedding_lookup(W, self.input_x2,name="embedded_chars_inp2")
                self.embedded_chars_inp2_expanded = tf.expand_dims(self.embedded_chars_inp2, axis=-1, name="embedded_chars_inp2_expanded") #conv2d needs 4d tensor of shape [batch, width(inp1_seq_len), height(embedding_size), channel(adding artifically here)].
 

        #blockA
        inp1_avgpooled_outputs_groupA = []
        inp1_minpooled_outputs_groupA = []
        inp1_maxpooled_outputs_groupA = []
        inp2_avgpooled_outputs_groupA = []
        inp2_minpooled_outputs_groupA = []
        inp2_maxpooled_outputs_groupA = []

        inp1_minpooled_outputs_groupB = []
        inp1_maxpooled_outputs_groupB = []
        inp2_minpooled_outputs_groupB = []
        inp2_maxpooled_outputs_groupB = []

        all_type_inp2_pools_groupa =[]
        all_type_inp1_pools_groupa =[]

        all_type_inp2_pools_groupb =[]
        all_type_inp1_pools_groupb =[]

        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("BlockA-%s" % filter_size):
                with tf.name_scope("embedded_chars_inp1_expanded"):
                    inp1_avgpooled, inp1_maxpooled, inp1_minpooled = self.groupa(filter_size, embedding_size, num_filters_A, self.embedded_chars_inp1_expanded,
                                                             inp1_sequence_length)
                # shape of inp1_avgpooled is (MB x Num_Filters_for_each_size)
                with tf.name_scope("embedded_chars_inp2_expanded"):
                    inp2_avgpooled, inp2_maxpooled, inp2_minpooled = self.groupa(filter_size, embedding_size, num_filters_A, self.embedded_chars_inp2_expanded,
                                                             inp2_sequence_length,reuse_flag = True)

                inp1_avgpooled_outputs_groupA.append(inp1_avgpooled) #shape is [types_of_filter_sizes x MB x Num_Filters_for_each_size]
                inp1_maxpooled_outputs_groupA.append(inp1_maxpooled)
                inp1_minpooled_outputs_groupA.append(inp1_minpooled)
                inp2_avgpooled_outputs_groupA.append(inp2_avgpooled)
                inp2_maxpooled_outputs_groupA.append(inp2_maxpooled)
                inp2_minpooled_outputs_groupA.append(inp2_minpooled)


            with tf.variable_scope("BlockB-%s" % filter_size):
                inp1_maxpool_perdimension = []
                inp1_minpool_perdimension = []
                inp2_maxpool_perdimension = []
                inp2_minpool_perdimension = []
                for embedding_idx in xrange(embedding_size):
                    embedded_chars_inp1_expanded_shape = tf.shape(self.embedded_chars_inp1_expanded)#.get_shape()
                    #self.embedded_chars_inp1_expanded = tf.Print(self.embedded_chars_inp1_expanded,[self.embedded_chars_inp1_expanded,"embedded_chars_inp1_expanded"])
                    shape0 = embedded_chars_inp1_expanded_shape[0]
                    shape1 = embedded_chars_inp1_expanded_shape[1]
                    shape3 = embedded_chars_inp1_expanded_shape[3]
                    embedding_slice_inp1 = tf.slice(self.embedded_chars_inp1_expanded,[0,0,embedding_idx,0],[shape0,shape1,1,shape3],"embedding_slice")
                    embedding_slice_inp2 = tf.slice(self.embedded_chars_inp2_expanded, [0, 0, embedding_idx, 0],[shape0, shape1, 1, shape3], "embedding_slice")
                    
                    with tf.variable_scope("Embedding_scope_%s" % embedding_idx):
                        _, inp1_maxpooled, inp1_minpooled = self.groupa(filter_size, 1, num_filters_B, embedding_slice_inp1, inp1_sequence_length)
                        _, inp2_maxpooled, inp2_minpooled = self.groupa(filter_size, 1, num_filters_B, embedding_slice_inp2, inp2_sequence_length,reuse_flag = True)
                        
                    inp1_maxpool_perdimension.append(inp1_maxpooled)  # [embed_dim x (MB x Num_filter)]
                    inp1_minpool_perdimension.append(inp1_minpooled)
                    
                    inp2_maxpool_perdimension.append(inp2_maxpooled)
                    inp2_minpool_perdimension.append(inp2_minpooled)

                inp1_maxpooled_outputs_groupB.append(tf.transpose(tf.stack(inp1_maxpool_perdimension),perm=[1,0,2], name ="transpose_inp1_maxpool_perdimension_102") )   #[types_of_filter_sizes x (MB x embed_dim x Num_filter)]
                inp1_minpooled_outputs_groupB.append(tf.transpose(tf.stack(inp1_minpool_perdimension),perm=[1,0,2]) )
                
                inp2_maxpooled_outputs_groupB.append(tf.transpose(tf.stack(inp2_maxpool_perdimension),perm=[1,0,2]) )
                inp2_minpooled_outputs_groupB.append(tf.transpose(tf.stack(inp2_minpool_perdimension),perm=[1,0,2]) )

        all_type_inp1_pools_groupa.append(inp1_avgpooled_outputs_groupA)
        all_type_inp1_pools_groupa.append(inp1_maxpooled_outputs_groupA) #shape is [ types_of_poolings(3)  x types_of_filter_sizes x MB x Num_Filters_for_each_size]
        all_type_inp1_pools_groupa.append(inp1_minpooled_outputs_groupA)

        all_type_inp2_pools_groupa.append(inp2_avgpooled_outputs_groupA)
        all_type_inp2_pools_groupa.append(inp2_maxpooled_outputs_groupA)
        all_type_inp2_pools_groupa.append(inp2_minpooled_outputs_groupA)



        all_type_inp1_pools_groupb.append(inp1_maxpooled_outputs_groupB)  # [types_of_poolings(2)  x types_of_filter_sizes x (MB x embed_dim x Num_filter)]
        all_type_inp1_pools_groupb.append(inp1_minpooled_outputs_groupB)

        all_type_inp2_pools_groupb.append(inp2_maxpooled_outputs_groupB)
        all_type_inp2_pools_groupb.append(inp2_minpooled_outputs_groupB)

        feah = self.Algo1(all_type_inp1_pools_groupa,all_type_inp2_pools_groupa)
        feaa,feab =self.Algo2(all_type_inp1_pools_groupa,all_type_inp2_pools_groupa,all_type_inp1_pools_groupb,all_type_inp2_pools_groupb)


        #fea = feah
        number_of_filter_windows = len(filter_sizes)
        fea_shape_1 =int( 
                    (3 * 2*num_filters_A)+  #here 3 is the num_pooling_ops
                    (3 *number_of_filter_windows * number_of_filter_windows * 3) + #here first 3 is the num_pooling_ops
                    (2 * number_of_filter_windows * num_filters_B * 3) #here 2 is the num_pooling_ops
                    )
 
        fea = tf.concat([feah,feaa,feab],-1)  # shape is MB x fea_shape_1
 
        weights = {
            'hidden1': tf.get_variable("hidden1_w",[fea_shape_1, hidden_num_units]),
            'hidden2': tf.get_variable("hidden2_w",[hidden_num_units, hidden_num_units]),
            'output': tf.get_variable("output_w",[hidden_num_units, output_num_units]),
        }

        biases = {
            'hidden1': tf.get_variable("hidden1_b",[hidden_num_units]),
            'hidden2': tf.get_variable("hidden2_b",[hidden_num_units]),
            'output': tf.get_variable("output_b",[output_num_units])
        }

        hidden_layer1 = tf.add(tf.matmul(fea, weights['hidden1']), biases['hidden1'])
        activated_hidden_layer1 = tf.tanh(hidden_layer1)
        hidden_layer2 = tf.add(tf.matmul(activated_hidden_layer1, weights['hidden2']), biases['hidden2'])
        activated_hidden_layer2 = tf.tanh(hidden_layer2)

        with tf.name_scope("output"):
            self.scores = tf.nn.xw_plus_b(activated_hidden_layer2, weights['output'], biases['output'], name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) 

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")



















