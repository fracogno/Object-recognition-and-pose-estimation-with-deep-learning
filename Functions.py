from PIL import Image
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
from decimal import Decimal


def normalize_img(img):
    
    for i in range(3): #3 channels between [0,1]
        min_val = np.min(img[:,:,i])
        max_val = np.max(img[:,:,i])

        img[:,:,i] -= min_val
        img[:,:,i] /= max_val

    return img


def get_images(path, categories, pose_file):
    img_list, classes, quaternions = [], [], []
    cl = 0
    
    for i in range(0, len(categories)):
     
        files_in_folder = glob.glob('./dataset/' + path + '/' + categories[i] + '/*.png')
        files_in_folder.sort(key = lambda x: int((x.split("/")[-1]).split(".")[0][len(path):]))
        for filename in files_in_folder:
            image = Image.open(filename)
            image = np.array(image)
            img_list.append(image)
            classes.append(i)
        
        count_line = 0
        file = open('./dataset/' + path + '/' + categories[i] + '/' + pose_file, "r")
        for line in file:
            if count_line % 2 != 0:
                quaternions.append(np.float64(line.split()))
            count_line += 1
        
        cl += 1
        
    assert(len(img_list) == len(classes) == len(quaternions))
    
    return np.array(img_list), np.array(classes), np.array(quaternions, dtype=np.dtype(Decimal))


def split_data(imgs, classes, quats, split_training, index_offset):
    assert(len(imgs) == len(classes) == len(quats))
    
    train_img, train_class, train_quat = [], [], [] 
    test_img, test_class, test_quat = [], [], []
    
    i_prec = 0
    for i in range(index_offset, len(imgs)+1, index_offset):
        temp_img = imgs[i_prec:i,:,:]
        temp_class = classes[i_prec:i]
        temp_quat = quats[i_prec:i,:]

        train_img.extend(temp_img[split_training])
        train_class.extend(temp_class[split_training])
        train_quat.extend(temp_quat[split_training])

        test_img.extend(np.delete(imgs[i_prec:i,:,:], split_training, axis = 0))
        test_class.extend(np.delete(classes[i_prec:i], split_training, axis = 0))
        test_quat.extend(np.delete(quats[i_prec:i,:], split_training, axis = 0))

        i_prec = i
        
    assert(len(train_img) + len(test_img) == len(imgs))

    return np.array(train_img), np.array(train_class), np.array(train_quat), \
            np.array(test_img), np.array(test_class), np.array(test_quat)


def batch_generator(train_img, train_class, train_quat, db_img, db_class, db_quat, batch_size):
    
    mask_random = np.random.choice(len(train_img), batch_size)

    #get anchors
    anchors_img = train_img[mask_random]
    anchors_class = train_class[mask_random]
    anchors_quat = train_quat[mask_random]
    
    #find puller
    puller_ind = []
    for i in range(0, batch_size):
        sel_indices = np.where(db_class == anchors_class[i])[0]

        min_angle = 181
        min_index = -1
        for j in range(0, len(sel_indices)):
            angle = quaternion_similarity(anchors_quat[i], db_quat[sel_indices[j]])

            if angle < min_angle:
                min_angle = angle
                min_index = sel_indices[j]
        puller_ind.append(min_index)            
            
    puller_img = db_img[puller_ind]
    puller_class = db_class[puller_ind]
    puller_quat = db_quat[puller_ind]

    #find pushers
    pusher_ind = []
    i = 0
    while len(pusher_ind) != batch_size:
        index = np.random.choice(len(db_img), 1)[0]
        
        if anchors_class[i] != db_class[index] or (anchors_class[i] == db_class[index] and quaternion_similarity(anchors_quat[i], db_quat[index]) > 91):
            pusher_ind.append(index)
            i += 1
    
    pusher_img = db_img[pusher_ind]
    pusher_class = db_class[pusher_ind]
    pusher_quat = db_quat[pusher_ind]
    
    #COMBINE ALL
    batch_img, batch_class, batch_quat = [], [], []
    for i in range(0, batch_size):
        batch_img.extend([anchors_img[i], puller_img[i], pusher_img[i]])
        batch_class.extend([anchors_class[i], puller_class[i], pusher_class[i]])
        batch_quat.extend([anchors_quat[i], puller_quat[i], pusher_quat[i]])
    
    return np.array(batch_img), np.array(batch_class), np.array(batch_quat)
    

def quaternion_similarity(q1, q2):
    # θ = arccos[ 2*⟨q1,q2⟩^2 − 1 ]
    return np.rad2deg(math.acos(2 * (np.clip((np.dot(q1, q2)), -1, +1) ** 2) - 1))
    
    
def plot_images(images):
    assert len(images) == 9
    
    labels = ["Anchor", "Puller", "Pusher"]
    
    # Figure with 3x3 subplots
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='binary')            
        ax.set_xlabel(labels[i % 3])
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.show()

    
##################TENSORFLOW######################

def new_weights(shape, name=None):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05), name=name)


def new_biases(length, name=None):
    return tf.Variable(tf.constant(0.05, shape=[length]), name=name)


def new_conv_layer(input, num_input_channels, filter_size, num_filters, pooling=True, name=None):  
    
    shape = [filter_size, filter_size, num_input_channels, num_filters] #as defined in tensorflow documentation
    
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)

    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding="VALID", name=name)
    layer += biases  # Add biases to each filter-channel

    layer = tf.nn.relu(layer)

    if pooling:
        layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    return layer


def flatten_layer(layer, name=None):
    layer_shape = layer.get_shape()  # [num_images, img_height, img_width, num_channels]

    num_features = layer_shape[1:4].num_elements() # The number of features is: img_height * img_width * num_channels
    
    layer_flat = tf.reshape(layer, [-1, num_features], name=name)  # [num_images, img_height * img_width * num_channels]

    return layer_flat, num_features


def new_fc_layer(input, num_inputs, num_outputs, relu=True, name=None): 

    weights = new_weights(shape=[num_inputs, num_outputs], name=name)
    biases = new_biases(length=num_outputs, name=name)

    layer = tf.matmul(input, weights) + biases

    if relu:
        layer = tf.nn.relu(layer)

    return layer


def total_loss(features, m = 0.01):
    return triplets_loss(features, m) + pairs_loss(features)
    
    
def triplets_loss(feature_desc, m = 0.01):
    batch_size = feature_desc.shape[0]
    
    diff_pos = tf.reduce_sum(tf.square(feature_desc[0:batch_size:3] - feature_desc[1:batch_size:3]), 1)
    diff_neg = tf.reduce_sum(tf.square(feature_desc[0:batch_size:3] - feature_desc[2:batch_size:3]), 1)

    loss = tf.maximum(0., 1 - (diff_neg / (diff_pos + m)))
    loss = tf.reduce_sum(loss)
    
    return loss


def pairs_loss(feature_desc):
    batch_size = feature_desc.shape[0]

    loss = tf.reduce_sum(tf.square(feature_desc[0:batch_size:3] - feature_desc[1:batch_size:3]), 1)
    loss = tf.reduce_sum(loss)
    
    return loss


def output_features(S_db, S_test, output_layer, loss, x, index_model):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, "./checkpoint/model" + str(index_model) +".ckpt") 

        descriptors_db = sess.run(output_layer, feed_dict={x: S_db})
        tmp_1 = sess.run(output_layer, feed_dict={x: S_test[:1200]})
        tmp_2 = sess.run(output_layer, feed_dict={x: S_test[1200:2400]})
        tmp_3 = sess.run(output_layer, feed_dict={x: S_test[2400:]})

    descriptors_test = np.concatenate((tmp_1, tmp_2, tmp_3), axis = 0)    
    
    assert(len(descriptors_test) == len(S_test))
    assert(len(S_db) == len(descriptors_db))
    
    return descriptors_db, descriptors_test
        
        
def matching_feature_map(feat_db, feat_test, db_classes, test_classes, db_quat, test_quat):
    indeces = []
    angles = [10.0, 20.0, 40.0, 180.0]
    angles_count = [0.0, 0.0, 0.0, 0.0]
    confusion_matrix = np.zeros((5,5))
    
    for i in range(len(feat_test)):

        min_dist = 1000000000.
        chosen_index = -1
        for j in range(len(feat_db)):
            dist = Euclidean_distance(feat_test[i], feat_db[j])

            if dist < min_dist:
                min_dist = dist
                chosen_index = j
        
        indeces.append(chosen_index)
        confusion_matrix[test_classes[i], db_classes[chosen_index]] += 1
        
        if db_classes[chosen_index] == test_classes[i]:
            quat_simil = quaternion_similarity(db_quat[chosen_index], test_quat[i])

            for k in range(len(angles)):
                if quat_simil <= angles[k]:
                    angles_count[k] += 1
    
    angles_count = np.array(angles_count)
    angles_count = (angles_count * 100.) / float(len(feat_test))
    
    return angles_count, indeces, confusion_matrix


def Euclidean_distance(f1, f2):
    return np.sqrt(np.sum(np.square(f1 - f2)))


def save_histogram(hist, index):
    assert(len(hist) == 4)
    
    angles = np.array(list(range(4)))
    hist_plot = plt.bar(angles, hist)
    
    plt.xticks(angles, ("10°", "20°", "40°", "180°"))

    plt.title('Iteration ' + str(index))
    plt.xlabel('Angle')
    plt.ylabel('Accuracy (%)')
    
    for rect in hist_plot:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.2f' % height, ha='center', va='bottom')
    
    plt.savefig('./histograms/' + str(index) + '.png')
    plt.clf()
    plt.cla()
    plt.close()
    
    
def save_plot_angles(histogram):
    
    histogram = np.array(histogram)

    indeces = np.array(list(range(0, len(histogram))))
    indeces *= 1000

    fig = plt.figure()

    ax1 = fig.add_subplot(221)
    ax1.plot(indeces, histogram[:,0], 'r-')
    ax1.set_title("Angle < 10°")

    ax2 = fig.add_subplot(222)
    ax2.plot(indeces, histogram[:,1], 'k-')
    ax2.set_title("Angle < 20°")

    ax3 = fig.add_subplot(223)
    ax3.plot(indeces, histogram[:,2], 'b-')
    ax3.set_title("Angle < 40°")

    ax4 = fig.add_subplot(224)
    ax4.plot(indeces, histogram[:,3], 'g-')
    ax4.set_title("Angle < 180°")

    plt.tight_layout()
    fig = plt.gcf()

    plt.savefig('./histograms/final_angles.png')    
   
    plt.clf()
    plt.cla()
    plt.close()