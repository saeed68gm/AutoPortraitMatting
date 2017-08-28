from __future__ import print_function
import tensorflow as tf
import numpy as np

import TensorflowUtils_plus as utils
#import read_MITSceneParsingData as scene_parsing
import datetime
#import BatchDatsetReader as dataset
from portrait_plus import BatchDatset, TestDataset
from PIL import Image
from six.moves import xrange
from scipy import misc
from tensorflow.python.framework import graph_util
from tensorflow.python.tools import freeze_graph

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "5", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 2
IMAGE_WIDTH = 600
IMAGE_HEIGHT = 800
train_size = 1529
test_size = 300

def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        if name in ['conv3_4', 'relu3_4', 'conv4_4', 'relu4_4', 'conv5_4', 'relu5_4']:
            continue
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if FLAGS.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    return net


def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])

    #processed_image = utils.process_image(image, mean_pixel)

    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, image)
        conv_final_layer = image_net["conv5_3"]

        pool5 = utils.max_pool_2x2(conv_final_layer)

        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        if FLAGS.debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        if FLAGS.debug:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

        expandDims = tf.expand_dims(annotation_pred, dim=3, name="expand_dims")

    return expandDims, conv_t3


def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


def main(argv=None):
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 6], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name="annotation")

    pred_annotation, logits = inference(image, keep_probability)
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(annotation, squeeze_dims=[3]), logits=logits, name="entropy")))
    tf.summary.scalar("entropy", loss)

    trainable_var = tf.trainable_variables()
    train_op = train(loss, trainable_var)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()
    '''
    print("Setting up image reader...")
    train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
    print(len(train_records))
    print(len(valid_records))

    print("Setting up dataset reader")
    image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
    if FLAGS.mode == 'train':
        train_dataset_reader = dataset.BatchDatset(train_records, image_options)
    validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)
    '''
    train_dataset_reader = BatchDatset('data/trainlist.mat')

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()
    print("Setting up summary writer")
    summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    #if FLAGS.mode == "train":
    itr = 0
    train_images, train_annotations = train_dataset_reader.next_batch()
    trloss = 0.0
    while len(train_annotations) > 0:
        #train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
        #print('==> batch data: ', train_images[0][100][100], '===', train_annotations[0][100][100])
        feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.5}
        summary_str, rloss =  sess.run([train_op, loss], feed_dict=feed_dict)
        trloss += rloss

        if itr % 2 == 0 and itr != 0:
            #train_loss, rpred = sess.run([loss, pred_annotation], feed_dict=feed_dict)
            print("Step: %d, Train_loss:%f" % (itr, trloss / 100))
            print("saving model at iteration #:", itr)
            saver.save(sess, FLAGS.logs_dir + "plus_model.ckpt", itr)
            trloss = 0.0
            tf.summary.scalar("loss", rloss)
            summary_writer.add_summary(summary_str, itr)

        #if itr % 10000 == 0 and itr > 0:
        '''
        valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
        valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations,
                                                       keep_probability: 1.0})
        print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))'''
        itr += 1
        # print("next batch : # %d", itr)
        train_images, train_annotations = train_dataset_reader.next_batch()

    '''elif FLAGS.mode == "visualize":
        valid_images, valid_annotations = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
        pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
                                                    keep_probability: 1.0})
        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)

        for itr in range(FLAGS.batch_size):
            utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5+itr))
            utils.save_image(valid_annotations[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5+itr))
            utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5+itr))
            print("Saved image: %d" % itr)'''

def freeze_and_store():
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    output_node_names = "ExpandDims"
    input_graph_name = "input_graph.pb"
    print ("graph node names", output_node_names)
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 6], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name="annotation")

    pred_annotation, logits = inference(image, keep_probability)
    sft = tf.nn.softmax(logits)
    test_dataset_reader = TestDataset('data/testlist.mat')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state("temp/")
        input_checkpoint = ckpt.model_checkpoint_path
        absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
        output_graph = absolute_model_folder + "/frozen_model.pb"
        saver = tf.train.Saver()
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored...")
        # We use a built-in TF helper to export variables to constants
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,# The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(),# The graph_def is used to retrieve the nodes
            output_node_names.split(",")# The output node names are used to select the usefull nodes
        )
        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))
        input_graph_def = graph.as_graph_def()
        tf.train.write_graph(input_graph_def, absolute_model_folder, input_graph_name)
        # print("using freeze_graph script")
        # input_graph_path = absolute_model_folder + "/input_graph.pb"
        # input_saver_def_path = ""
        # input_binary = False
        # checkpoint_path = ckpt.model_checkpoint_path
        # restore_op_name = "save/restore_all"
        # filename_tensor_name = "save/Const"
        # output_graph_path = output_graph
        # #output_graph_path = os.path.join(self.get_temp_dir(), output_graph_name)
        # clear_devices = False
        # freeze_graph.freeze_graph(
        #     input_graph_path, input_saver_def_path,
        #     input_binary, checkpoint_path, output_node_names,
        #     restore_op_name, filename_tensor_name, output_graph_path, clear_devices, "")
    print("saving graph to file successful")

def pred():
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 6], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name="annotation")

    pred_annotation, logits = inference(image, keep_probability)
    sft = tf.nn.softmax(logits)
    test_dataset_reader = TestDataset('data/testlist.mat')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        saver = tf.train.Saver()
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored...")
        itr = 0
        test_images, test_annotations, test_orgs = test_dataset_reader.next_batch()
        #print('getting', test_annotations[0, 200:210, 200:210])
        while len(test_annotations) > 0:
            print("iteration #: %d", itr)
            if itr < 22:
                test_images, test_annotations, test_orgs = test_dataset_reader.next_batch()
                itr += 1
            elif itr > 22:
                break
            feed_dict = {image: test_images, annotation: test_annotations, keep_probability: 0.5}
            rsft, pred_ann = sess.run([sft, pred_annotation], feed_dict=feed_dict)
            print(rsft.shape)
            _, h, w, _ = rsft.shape
            preds = np.zeros((h, w, 1), dtype=np.float)
            for i in range(h):
                for j in range(w):
                    if rsft[0][i][j][0] < 0.1:
                        preds[i][j][0] = 1.0
                    elif rsft[0][i][j][0] < 0.9:
                        preds[i][j][0] = 0.5
                    else:
                        preds[i][j]  = 0.0
            org0_im = Image.fromarray(np.uint8(test_orgs[0]))
            org0_im.save('res/org' + str(itr) + '.jpg')
            save_alpha_img(test_orgs[0], test_annotations[0], 'res/ann' + str(itr))
            save_alpha_img(test_orgs[0], preds, 'res/trimap' + str(itr))
            save_alpha_img(test_orgs[0], pred_ann[0], 'res/pre' + str(itr))
            test_images, test_annotations, test_orgs = test_dataset_reader.next_batch()
            itr += 1

def save_alpha_img(org, mat, name):
    w, h = mat.shape[0], mat.shape[1]
    #print(mat[200:210, 200:210])
    rmat = np.reshape(mat, (w, h))
    amat = np.zeros((w, h, 4), dtype=np.int)
    amat[:, :, 3] = np.round(rmat * 1000)
    amat[:, :, 0:3] = org
    #print(amat[200:205, 200:205])
    #im = Image.fromarray(np.uint8(amat))
    #im.save(name + '.png')
    misc.imsave(name + '.png', amat)

if __name__ == "__main__":
    #tf.app.run()
    #pred()
    freeze_and_store()
