from __future__ import print_function
import tensorflow as tf
import numpy as np

import inference as inf
import TensorflowUtils_plus as utils
import datetime
from eval_plus import TestDataset
from PIL import Image
from six.moves import xrange
from scipy import misc

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'


CKPT_DIR = "logs/"
MODEL_DIR = "logs/frozen_model.pb"
INPUT_MAP_DIR = "logs/input_graph.pb"
IMAGE_WIDTH = 600
IMAGE_HEIGHT = 800
NUM_OF_CLASSESS = 2

def print_layers(graph_def):
    print("printing layers ...")
    nodes = [n.name for n in graph_def.node]
    for i in range(0, len(nodes)):
        print(nodes[i])
    print("printing layers done!")

def load_graph(frozen_graph_filename, input_map_filename=None):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    if (input_map_filename == None):
        input_map = None
    else:
        with tf.gfile.GFile(input_map_filename, "rb") as input_map_file:
            input_map = tf.GraphDef()
            input_map.ParseFromString(input_map_file.read())
            print("input map read")

    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=input_map,
            return_elements=None,
            name="",
            op_dict=None,
            producer_op_list=None
        )
    print_layers(graph_def);
    return graph

def save_alpha_img(org, mat, name):
    w, h = mat.shape[0], mat.shape[1]
    rmat = np.reshape(mat, (w, h))
    amat = np.zeros((w, h, 4), dtype=np.int)
    amat[:, :, 3] = np.round(rmat * 1000)
    amat[:, :, 0:3] = org
    misc.imsave(name + '.png', amat)

def pred():
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 6], name="input_image")

    pred_annotation, logits, conv_final_layer, relu5_3 = inf.inference(image, keep_probability)
    sft = tf.nn.softmax(logits)
    test_dataset_reader = TestDataset('data/list.mat')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        saver = tf.train.Saver()
        print ("logs dir : ", FLAGS.logs_dir)
        print ("ckpt : ", ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored...")
        graph_def = tf.get_default_graph().as_graph_def()
        import pdb; pdb.set_trace()  # breakpoint a11629bd //
        print_layers(graph_def)
        itr = 0
        test_images, test_orgs = test_dataset_reader.next_batch()
        while len(test_images) > 0:
            print("iteration #: %d", itr)
            if itr > 9:
                break
            feed_dict = {image: test_images, keep_probability: 0.5}
            rsft, pred_ann, conv_layer, relu53 = sess.run([sft, pred_annotation, conv_final_layer, relu5_3], feed_dict=feed_dict)
            for i in range(0, 512):
                A = relu53[0][:,:,i]
                B = conv_layer[0][:,:,i]
                misc.imsave("debug/relu_" + str(i) + ".png", A)
                misc.imsave("debug/conv_" + str(i)+ ".png", B)
            import pdb; pdb.set_trace()  # breakpoint 95fdafc3 //
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
            print("saving image : ", itr)
            org0_im = Image.fromarray(np.uint8(test_orgs[0]))
            org0_im.save('res/org' + str(itr) + '.jpg')
            save_alpha_img(test_orgs[0], preds, 'res/trimap' + str(itr))
            save_alpha_img(test_orgs[0], pred_ann[0], 'res/pre' + str(itr))
            test_images, test_orgs = test_dataset_reader.next_batch()
            itr += 1

def pred_with_frozen():
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 6], name="input_image")
    # with tf.name_scope('input_reshape'):
    #     image_shaped_input = tf.reshape(image, [-1, 28, 28, 1])
    #     tf.summary.image("input_image", image_shaped_input, 10)

    expandDims, logits, conv_final_layer, relu5_3 = inf.inference(image, keep_probability, False, True)
    #tf.summary.image("output_image", expandDims)
    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    sft = tf.nn.softmax(logits)
    test_dataset_reader = TestDataset('data/list.mat')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        print ("logs dir : ", FLAGS.logs_dir)
        graph = load_graph(MODEL_DIR)
        #print layer names
        for op in graph.get_operations():
            print(op.name)
        summary_writer = tf.summary.FileWriter("temp", sess.graph)

        itr = 0
        test_images, test_orgs = test_dataset_reader.next_batch()
        while len(test_images) > 0:
            print("iteration #: %d", itr)
            if itr > 9:
                break
            feed_dict = {image: test_images, keep_probability: 0.5}
            rsft, expand_dims, conv_layer, relu53 = sess.run([sft, expandDims, conv_final_layer, relu5_3], feed_dict=feed_dict)
            #summary_writer.add_summary(summary_str, itr)
            print(rsft.shape)
            print("annotation_pred : ")
            #print(annotation_pred.shape)
            for i in range(0, 512):
                A = relu53[0][:,:,i]
                B = conv_layer[0][:,:,i]
                misc.imsave("debug/relu_" + str(i) + "_frozen.png", A)
                misc.imsave("debug/conv_" + str(i)+ "_frozen.png", B)
            import pdb; pdb.set_trace()  # breakpoint 0f341756 //
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
            print("saving image : ", itr)
            org0_im = Image.fromarray(np.uint8(test_orgs[0]))
            #ann_im = Image.fromarray(np.uint8(annotation_pred))
            org0_im.save('res/org' + str(itr) + '.jpg')
            #ann_im.save('res/ann_' + str(itr) + '.jpg')
            save_alpha_img(test_orgs[0], expand_dims[0], 'res/pre' + str(itr))
            test_images, test_orgs = test_dataset_reader.next_batch()
            itr += 1
        summary_writer.close()

def main(argv=None):
  print("welcome!")
  pred_with_frozen()
  #pred();

if __name__ == "__main__":
  main()
