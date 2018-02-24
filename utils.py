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

def makebatches(batch_size, img_features, train_data):
    num_train = train_data['question'].shape[0]
    while(true):
    indices = np.random.random_integers(0, num_train-1, batch_size)
    current_image_list = train_data['img_list'][indices]

    if !len(img_features) in current_image_list:
        break

    current_images = img_features[current_image_list,:]

    current_questions = train_data['question'][indices,:]
    
    current_answers = train_data['answers'][indices]

    batch = {}
    batch["questions"] = current_questions
    batch["answers"] = current_answers
    batch["images"] = current_images
    #print "Returning batch", batch
    return batch

def right_align(seq, lengths):
    v = np.zeros(np.shape(seq))
    N = np.shape(seq)[1]
    for i in range(np.shape(seq)[0]):
        v[i][N-lengths[i]:N-1]=seq[i][0:lengths[i]-1]
    return v
