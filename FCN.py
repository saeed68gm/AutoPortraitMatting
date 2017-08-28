from __future__ import print_function
import tensorflow as tf
import numpy as np

import TensorflowUtils as utils
#import read_MITSceneParsingData as scene_parsing
import datetime
#import BatchDatsetReader as dataset
from portrait import BatchDatset, TestDataset
#from eval_plus import TestDataset
from six.moves import xrange
from PIL import Image
from scipy import misc
from tensorflow.core.framework import graph_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from tensorflow.python.summary import summary
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from tensorflow.python.saved_model import builder as saved_model_builder


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "fcn_logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MODEL_DIR = FLAGS.logs_dir + "tf_freeze_FCN.pb"

MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 2
IMAGE_WIDTH = 600
IMAGE_HEIGHT = 800

def print_layers(graph_def):
    print("printing layers ...")
    nodes = [n.name for n in graph_def.node]
    for i in range(0, len(nodes)):
        print(nodes[i])
    print("printing layers done!")


def load_graph(frozen_graph_filename, input_map_filename=None):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    # if (input_map_filename == None):
    #     input_map = None
    # else:
    #     with tf.gfile.GFile(input_map_filename, "rb") as input_map_file:
    #         input_map = tf.GraphDef()
    #         input_map.ParseFromString(input_map_file.read())
    #         print("input map read")

    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph

    with tf.Graph().as_default() as graph:

        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="",
            op_dict=None,
            producer_op_list=None
        )
    print_layers(graph_def);
    return graph

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


def inference(image):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    keep_prob = 0.5
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
    #keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name="annotation")

    pred_annotation, logits = inference(image)
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(annotation, squeeze_dims=[3]), logits=logits, name="entropy")))
    tf.summary.scalar("entropy", loss)

    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
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
    train_dataset_reader = BatchDatset('data/trainlist_fcn.mat')

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    #if FLAGS.mode == "train":
    itr = 0
    with tf.device("/gpu:0"):
        train_images, train_annotations = train_dataset_reader.next_batch()
        while len(train_annotations) > 0:
            #train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
            #print('==> batch data: ', train_images[0][100][100], '===', train_annotations[0][100][100])
            feed_dict = {image: train_images, annotation: train_annotations}

            sess.run(train_op, feed_dict=feed_dict)
            train_loss, summary_str, rpred = sess.run([loss, summary_op, pred_annotation], feed_dict=feed_dict)
            if itr % 400 == 0:
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                summary_writer.add_summary(summary_str, itr)
                print(np.sum(rpred))
                print('=============')
                print(np.sum(train_annotations))
                print('------------>>>')
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)

            #if itr % 10000 == 0 and itr > 0:
            '''
            valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
            valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations,
                                                           keep_probability: 1.0})
            print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))'''

            itr += 1
            train_images, train_annotations = train_dataset_reader.next_batch()

    saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)
    output_node_names = "inference/expand_dims"
    output_graph_def = graph_util.convert_variables_to_constants(
        sess,# The session is used to retrieve the weights
        tf.get_default_graph().as_graph_def(),# The graph_def is used to retrieve the nodes
        output_node_names.split(",")# The output node names are used to select the usefull nodes
    )
    # Finally we serialize and dump the output graph to the filesystem
    output_graph = FLAGS.logs_dir + "/frozen_model_FCN.pb"
    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))
    input_graph_def = graph.as_graph_def()
    tf.train.write_graph(input_graph_def, FLAGS.logs_dir, input_graph_name)
    print("saving graph to file successful")
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


def pred():
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name="annotation")

    pred_annotation, logits = inference(image)
    test_dataset_reader = TestDataset('data/list.mat')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        saver = tf.train.Saver()
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored...")
        itr = 0
        test_images, test_orgs = test_dataset_reader.next_batch()
        # A = misc.imread("data/stylized1.png")
        # read_im = A[:,:,:3]
        # test_images = read_im.reshape(1, 800, 600, 3)
        #print('getting', test_annotations[0, 200:210, 200:210])
        while len(test_images) > 0:
            print("iteration #: %d", itr)
            if itr > 9:
                break
            feed_dict = {image: test_images}
            preds = sess.run(pred_annotation, feed_dict=feed_dict)
            print("saving image : ", itr)
            org0_im = Image.fromarray(np.uint8(test_orgs[0]))
            org0_im.save('res/org' + str(itr) + '.jpg')
            save_alpha_img(test_orgs[0], preds[0], 'res/pre' + str(itr))
            test_images, test_orgs = test_dataset_reader.next_batch()
            itr += 1


def pred_with_frozen():
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name="input_image")
    # annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name="annotation")
    # with tf.name_scope('input_reshape'):
    #     image_shaped_input = tf.reshape(image, [-1, 28, 28, 1])
    #     tf.summary.image("input_image", image_shaped_input, 10)

    pred_annotation, logits = inference(image)
    #tf.summary.image("output_image", expandDims)
    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    sft = tf.nn.softmax(logits)
    test_dataset_reader = TestDataset('data/list.mat')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        print ("logs dir : ", FLAGS.logs_dir)
        # graph = load_graph("test_logs/test_freeze.pb")
        # #print layer names
        graph = tf.get_default_graph()
        org_graph_def = graph.as_graph_def()
        with gfile.FastGFile("fcn_logs/tf_freeze_FCN.pb", "rb") as f:
          graph_def = graph_pb2.GraphDef()
          graph_def.ParseFromString(f.read())
          importer.import_graph_def(graph_def)
          org_graph_def = graph_def

        for op in graph.get_operations():
            print(op.name)
        summary_writer = tf.summary.FileWriter("temp", sess.graph)

        itr = 0
        test_images, test_orgs = test_dataset_reader.next_batch()
        while len(test_images) > 0:
            print("iteration #: %d", itr)
            if itr > 9:
                break
            feed_dict = {image: test_images}
            preds = sess.run(pred_annotation, feed_dict=feed_dict)
            print("saving image : ", itr)
            org0_im = Image.fromarray(np.uint8(test_orgs[0]))
            org0_im.save('res/org' + str(itr) + '.jpg')
            save_alpha_img(test_orgs[0], preds[0], 'res/pre' + str(itr))
            test_images, test_orgs = test_dataset_reader.next_batch()
            itr += 1
        summary_writer.close()

def freeze_and_store():
    output_node_names = "inference/expand_dims"
    print ("graph node names", output_node_names)
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name="annotation")

    pred_annotation, logits = inference(image)
    sft = tf.nn.softmax(logits)
    test_dataset_reader = TestDataset('data/list.mat')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.70)
    #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess = tf.Session()
    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    input_checkpoint = ckpt.model_checkpoint_path
    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_folder + "/frozen_model_FCN.pb"
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    test_images, test_orgs = test_dataset_reader.next_batch()
    feed_dict = {image: test_images}
    preds = sess.run(pred_annotation, feed_dict=feed_dict)
    save_alpha_img(test_orgs[0], preds[0], 'res/chkpt1')
    export_path = "export/model_v0"
    print("Exporting trained model to", export_path)
    builder = saved_model_builder.SavedModelBuilder(export_path)
    #TODO: update with correct arguments
    tensor_info_x = tf.saved_model.utils.build_tensor_info(image)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(annotation)
    prediction_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'input_image': tensor_info_x},
        outputs={'annotation': tensor_info_y},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

    builder.add_meta_graph_and_variables(
      sess, "serve",
      signature_def_map={
           'predict_images':
               prediction_signature
      },
      legacy_init_op=legacy_init_op)
    builder.save()
    # graph = tf.get_default_graph()
    # input_graph_def = graph.as_graph_def()
    # input_graph_name = "input_graph.pb"
    # print_layers(graph.as_graph_def())
    # graph_io.write_graph(sess.graph, FLAGS.logs_dir, input_graph_name)
    # for i in range(0, 512):
    #     A = relu53[0][:,:,i]
    #     B = conv_layer[0][:,:,i]
    #     misc.imsave("debug/relu_" + str(i) + "_restored.png", A)
    #     misc.imsave("debug/conv_" + str(i)+ "_restored.png", B)
    # input_graph_def = graph.as_graph_def()
    # input_graph_name = "input_graph.pb"
    # # We use a built-in TF helper to export variables to constants
    # output_graph_def = graph_util.convert_variables_to_constants(
    #     sess,# The session is used to retrieve the weights
    #     tf.get_default_graph().as_graph_def(),# The graph_def is used to retrieve the nodes
    #     output_node_names.split(",")# The output node names are used to select the usefull nodes
    # )
    # # Finally we serialize and dump the output graph to the filesystem
    # with tf.gfile.GFile(output_graph, "wb") as f:
    #     f.write(output_graph_def.SerializeToString())
    # print("%d ops in the final graph." % len(output_graph_def.node))
    # input_graph_def = graph.as_graph_def()
    # tf.train.write_graph(input_graph_def, absolute_model_folder, input_graph_name)

    # graph = load_graph(MODEL_DIR)
    # rsft, expand_dims = sess.run([sft, expandDims], feed_dict=feed_dict)
    # for i in range(0, 512):
    #     A = relu53[0][:,:,i]
    #     B = conv_layer[0][:,:,i]
    #     misc.imsave("debug/relu_" + str(i) + "_frozen.png", A)
    #     misc.imsave("debug/conv_" + str(i)+ "_frozen.png", B)
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

def save_bin(mat, name):
    with open(name, 'wb') as file:
        mat.tofile(file)

def save_alpha_img(org, mat, name):
    w, h, _ = mat.shape
    #print(mat[200:210, 200:210])
    #save_bin(mat, "binary_out.bin")
    rmat = np.reshape(mat, (w, h))
    amat = np.zeros((w, h, 4), dtype=np.int)
    amat[:, :, 3] = rmat * 1000
    amat[:, :, 0:3] = org
    print(amat[200:205, 200:205])
    #im = Image.fromarray(np.uint8(amat))
    #im.save(name + '.png')
    misc.imsave(name + '.png', amat)

if __name__ == "__main__":
    #tf.app.run()
    #pred()
    freeze_and_store()
    #pred_with_frozen()
