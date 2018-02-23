import json
import h5py
import numpy as np

def get_data(config):

    dataset = {}
    train_data = {}
    # load json file
    print('loading json file...')
    with open(config.input_json) as data_file:
        data = json.load(data_file)

    for key in data.keys():
        dataset[key] = data[key]

    # load image feature
    print('loading image features...')
    with h5py.File(config.input_image_h5,'r') as hf:
        # -----0~82459------
        tem = hf.get('images_train')
        image_features = np.array(tem)

    # load h5 file
    print('loading h5 file...')
    with h5py.File(config.input_ques_h5,'r') as hf:
        
        # total number of training data is 215375
        # question is (26, )
        tem = hf.get('ques_train')
        train_data['question'] = np.array(tem)-1
        
        # max length is 23
        tem = hf.get('ques_length_train')
        train_data['length_q'] = np.array(tem)
        
        # total 82460 img
        tem = hf.get('img_pos_train')
	   
        # convert into 0~82459
        train_data['img_list'] = np.array(tem)-1
       
        # answer is 1~1000
        tem = hf.get('answers')
        train_data['answers'] = np.array(tem)-1

    print('Aligning question')
    train_data['question'] = right_align(train_data['question'], train_data['length_q'])

    print('Normalizing image feature')
    if config.img_norm:
        tem = np.sqrt(np.sum(np.multiply(image_features, image_features), axis=1))
        image_features = np.divide(image_features, np.transpose(np.tile(tem,(config.image_size,1))))

    return dataset, image_features, train_data

def get_data_test(config):
    dataset = {}
    test_data = {}
    # load json file
    print('loading json file...')
    with open(input_json) as data_file:
        data = json.load(data_file)
    
    for key in data.keys():
        dataset[key] = data[key]

    # load image feature
    print('loading image features...')
    with h5py.File(config.input_img_h5,'r') as hf:
        tem = hf.get('images_test')
        image_features = np.array(tem)

    # load h5 file
    print('loading h5 file...')
    with h5py.File(config.input_ques_h5,'r') as hf:
        # total number of training data is 215375
        # question is (26, )
        tem = hf.get('ques_test')
        test_data['question'] = np.array(tem)-1
        # max length is 23
        tem = hf.get('ques_length_test')
        test_data['length_q'] = np.array(tem)
        # total 82460 img
        tem = hf.get('img_pos_test')
        # convert into 0~82459
        test_data['img_list'] = np.array(tem)-1
        # quiestion id
        tem = hf.get('question_id_test')
        test_data['ques_id'] = np.array(tem)

    # MC_answer_test
    tem = hf.get('MC_ans_test')
    test_data['MC_ans_test'] = np.array(tem)


    print('question aligning')
    test_data['question'] = right_align(test_data['question'], test_data['length_q'])

    print('Normalizing image feature')
    if config.img_norm:
        tem = np.sqrt(np.sum(np.multiply(image_features, image_features), axis=1))
        image_features = np.divide(image_features, np.transpose(np.tile(tem,(config.image_size, 1))))

    return dataset, image_features, test_data

def get_minibatches(data, minibatch_size, shuffle=True):
    """
    Iterates through the provided data one minibatch at at time. You can use this function to
    iterate through data in minibatches as follows:

        for inputs_minibatch in get_minibatches(inputs, minibatch_size):
            ...

    Or with multiple data sources:

        for inputs_minibatch, labels_minibatch in get_minibatches([inputs, labels], minibatch_size):
            ...

    Args:
        data: there are two possible values:
            - a list or numpy array
            - a list where each element is either a list or numpy array
        minibatch_size: the maximum number of items in a minibatch
        shuffle: whether to randomize the order of returned data
    Returns:
        minibatches: the return value depends on data:
            - If data is a list/array it yields the next minibatch of data.
            - If data a list of lists/arrays it returns the next minibatch of each element in the
              list. This can be used to iterate through multiple data sources
              (e.g., features and labels) at the same time.

    """
    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
        yield [minibatch(d, minibatch_indices) for d in data] if list_data \
            else minibatch(data, minibatch_indices)


def minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]

def minibatches(data, batch_size, shuffle=True):
    batches = [np.array(col) for col in zip(*data)]
    return get_minibatches(batches, batch_size, shuffle)

def right_align(seq, lengths):
    v = np.zeros(np.shape(seq))
    N = np.shape(seq)[1]
    for i in range(np.shape(seq)[0]):
        v[i][N-lengths[i]:N-1]=seq[i][0:lengths[i]-1]
    return v