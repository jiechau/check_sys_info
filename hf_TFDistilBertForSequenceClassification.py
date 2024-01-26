is_colab = False
is_rog = True # that C:/ problem， and gram problem

# vars
LOOP_NUM = 3
#LOOP_NUM = 1_000_000 #  無限次 就給 100_000
#EPOCHS = 100
EPOCHS = 1
DATASET_NUM = 10 #
#DATASET_NUM = 10_000 # 1000筆 22秒
#DATASET_NUM = -1 # 全部
TRAIN_SET_RATIO = 0.99
BATCH_SIZE = 8 # 那台 rog 只能用 8，(distilbert 只能用 4)
LEARNING_RATE = 1e-5 # 0.01
TEST_size = 0.01
MAX_SEQ_LEN = 128
num_classes = 2_000

if is_colab == False and is_rog == True:
    LOOP_NUM = 1_000_000 #  無限次 就給 1_000_000
    EPOCHS = 1
    DATASET_NUM = 10_000 # 22 分鐘存一次
    #DATASET_NUM = -1 # 全部
    BATCH_SIZE = 8

is_first_time_create_model = False
#is_first_time_create_model = True

#pretrained_model_name = 'bert-base-chinese'
pretrained_model_name = 'distilbert-base-multilingual-cased'
save_model_suffix = '-tf-keras'

import os
colab_dir = '/tmp/Colab_Notebooks'
HF_HOME = '/tmp/transformers'
tmp_dir = '/tmp' # save tmp

# colab_dir
os.environ["colab_dir"] = colab_dir
# HF cache
os.environ["HF_HOME"] = HF_HOME
os.makedirs(HF_HOME, exist_ok=True)
# saved_model
dir_all_model = colab_dir + '/_saved_model'
os.environ["dir_all_model"] = dir_all_model
# _saved_data
dir_all_data = colab_dir + '/_saved_data'
os.environ["dir_all_data"] = dir_all_data

#
dir_save_data = dir_all_data + '/friDay_data'
dir_saved_model = dir_all_model + '/_saved_model_friDay'
import pandas as pd
# df_all
pickle_location = dir_save_data + '/df_items_content_details_cate.pkl'
df_all = pd.read_pickle(pickle_location)
print('done', str(len(df_all)))
# df_all
pickle_location = dir_save_data + '/catg_content.pkl'
df_catg = pd.read_pickle(pickle_location)
## 這裡忘記改了
df_catg['cate_id'] = df_catg['level_b'].str.extract('(\d+)').astype(int)
print('done', str(len(df_catg)))
# max cate_id
print('max cate_id',str(df_all.cate_id.max()))


# load df data
df_all_tmp = df_all[['name', 'cate_id']].copy()
df_all_tmp.columns = ['product_name', 'category']
# dataset
from sklearn.model_selection import train_test_split
df_all_train, df_all_test = train_test_split(df_all_tmp, test_size=(1-TRAIN_SET_RATIO))
#DATASET_NUM = 100 # 1000筆 22秒
#DATASET_NUM = -1 # 全部

def get_chunk():
    if DATASET_NUM != -1 and DATASET_NUM < len(df_all_tmp):
        n = int(DATASET_NUM * TRAIN_SET_RATIO)
        df_all_train_chunks = [df_all_train[i:i+n] for i in range(0,df_all_train.shape[0],n)]
        m = DATASET_NUM - n
        df_all_test_chunks = [df_all_test[i:i+m] for i in range(0,df_all_test.shape[0],m)]
        # 接下來，只好決定怎麼選
        # 理論上 len(df_all_train_chunks) 和 len(df_all_test_chunks) 是一樣的
        import random
        random_idx = random.randrange(len(df_all_train_chunks))
        df_train_data = df_all_train_chunks[random_idx]
        df_test_data = df_all_test_chunks[random_idx]
    else:
        df_train_data = df_all_train
        df_test_data = df_all_test
    print('dataset', len(df_train_data), len(df_test_data))
    return (df_train_data, df_test_data)










if pretrained_model_name == 'bert-base-chinese':
    save_model_dir_name = 'bert-base-chinese'
    from transformers import BertTokenizer, TFBertForSequenceClassification
    max_len = MAX_SEQ_LEN
    def get_tokenizer_new():
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        return tokenizer
    def get_tokenizer_old(model_path):
        tokenizer = BertTokenizer.from_pretrained(model_path)
        return tokenizer
    def get_model_new(num_classes):
        model = TFBertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=num_classes)
        return model
    def get_model_old(model_path):
        model = TFBertForSequenceClassification.from_pretrained(model_path)
        return model

if pretrained_model_name == 'distilbert-base-multilingual-cased':
    save_model_dir_name = 'distilbert-base-multilingual-cased'
    from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
    max_len = MAX_SEQ_LEN
    def get_tokenizer_new():
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
        return tokenizer
    def get_tokenizer_old(model_path):
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        return tokenizer
    def get_model_new(num_classes):
        model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-multilingual-cased', num_labels=num_classes)
        return model
    def get_model_old(model_path):
        model = TFDistilBertForSequenceClassification.from_pretrained(model_path)
        return model






# load model

model_save_h5_dir = dir_all_model + '/_saved_model_friDay/friDay_category_id' + '/' + save_model_dir_name # don't end '.h5'
model_save_h5_dir = model_save_h5_dir + save_model_suffix # don't end '.h5'
os.environ["model_save_h5_dir"] = model_save_h5_dir
os.makedirs(model_save_h5_dir, exist_ok=True)
print('------------------------------------------------------------------')
print(model_save_h5_dir)
print('------------------------------------------------------------------')
bb = '''
Saving the model to HDF5 format requires the model to be a Functional model or a Sequential model. It does not work for subclassed models, because such models are defined via the body of a Python method, which isn't safely serializable. Consider saving to the Tensorflow SavedModel format (by setting save_format="tf") or using `save_weights`.
ensure that the file path you are saving to does not end with .h5 as this extension is typically associated with HDF5 format.
'''

import tensorflow as tf
if is_first_time_create_model:
    # 1st time to create (ONLY '1st' !!!!!)
    # 第一次會一直寫入新的，要小心
    model = get_model_new(num_classes)
    tokenizer = get_tokenizer_new()
    tokenizer.save_pretrained(model_save_h5_dir)
    print('------------------------------------------------------------------')
    print('new new new new new new new new new new new')
else:
    # 這個部分暫時還是要，以後再刪掉
    tokenizer = get_tokenizer_new()
    tokenizer.save_pretrained(model_save_h5_dir)
    # load old one
    model = get_model_old(model_save_h5_dir)
    tokenizer = get_tokenizer_old(model_save_h5_dir)
    # model = tf.keras.models.load_model(model_save_h5_dir)
    print('------------------------------------------------------------------')
    print('old old old old old old old old old old old')
#
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
# compile
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])



print(model.summary())



# 610_012, 11 分鐘
import tensorflow as tf

import pandas as pd
from sklearn.model_selection import train_test_split

def convert_example_to_feature(review):
    return tokenizer.encode_plus(review,
                                 add_special_tokens=True,
                                 max_length=MAX_SEQ_LEN,
                                 padding='max_length',
                                 #pad_to_max_length=True,
                                 ##return_token_type_ids=True,  # 添加这一行
                                 return_attention_mask=True)

#def map_example_to_dict(input_ids, token_type_ids, attention_masks, label):
def map_example_to_dict(input_ids, attention_masks, label):
    return {
        "input_ids": input_ids,
        ##"token_type_ids": token_type_ids,
        "attention_mask": attention_masks,
    }, label

def encode_examples(ds, limit=-1):
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []

    if (limit > 0):
        ds = ds.take(limit)

    for index, row in ds.iterrows():
        bert_input = convert_example_to_feature(row['product_name'])
        input_ids_list.append(bert_input['input_ids'])
        ##token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append(row['category'])
    ##return tf.data.Dataset.from_tensor_slices((input_ids_list, token_type_ids_list, attention_mask_list, label_list)).map(map_example_to_dict)
    return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, label_list)).map(map_example_to_dict)

# dataset
#df_train_data, df_test_data = train_test_split(df, test_size=(1-TRAIN_SET_RATIO))

# 這裡
def get_dataset():

    df_train_data, df_test_data = get_chunk()

    train_data = encode_examples(df_train_data).shuffle(10000).batch(BATCH_SIZE)
    test_data = encode_examples(df_test_data).batch(BATCH_SIZE)

    # shuffle = 10_000 is a common choice

    #train_data = encode_examples(df_train_data).shuffle(10000).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    #test_data = encode_examples(df_test_data).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

    return (train_data, test_data)

# 自己 trigger 一次
# 610_012, 11 分鐘
train_data, test_data = get_dataset()



# train the model
from datetime import datetime,timedelta
_t1 = datetime.now()
def msg_time():
    global _t1
    _t2 = datetime.now()
    time_diff = (_t2 - _t1).total_seconds()
    time_diff_float = round(time_diff / 60, 2)
    msg_all = _t2.strftime("%Y-%m-%d %H:%M:%S") + ' ' + str(time_diff_float) + ' min'
    _t1 = _t2
    return msg_all
    #return

import tensorflow as tf
#from transformers import TFBertForSequenceClassification
# 要將 Hugging Face Transformers 的 save_pretrained 方法整合到 Keras 的回調函數中，
# 您需要創建一個自定義的回調。
# 這個自定義回調將在每個 epoch 結束時調用 save_pretrained 方法來保存模型。
class TransformersCheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_path, save_freq=1):
        super(TransformersCheckpointCallback, self).__init__()
        self.save_path = save_path
        self.save_freq = save_freq

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
        #if (epoch + 1) % 10 == 0:
            #save_path = os.path.join(self.save_path, f"epoch_{epoch+1}")
            #os.makedirs(save_path, exist_ok=True)
            self.model.save_pretrained(self.save_path)
            #print(msg_time(), str(epoch))

#from tensorflow.keras.callbacks import ModelCheckpoint



print(msg_time(), 'start')
#if True:
#while True:
for iii in range(LOOP_NUM):
    # dataset
    if DATASET_NUM != -1:
        train_data, test_data = get_dataset()
    # train, use callback
    ##checkpoint_callback = TransformersCheckpointCallback(model_save_h5_dir)
    ##history = model.fit(train_data, epochs=EPOCHS, validation_data=test_data, callbacks=[checkpoint_callback])
    # train, manually
    history = model.fit(train_data, epochs=EPOCHS, validation_data=test_data)
    if ((iii + 1) %  10) == 0:
        model.save_pretrained(model_save_h5_dir)
        print('s' * 80)

    #print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(msg_time(), 'loop:' + str(iii))
    # save model
    #model.save(model_save_h5_file)











