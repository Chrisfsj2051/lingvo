import tensorflow as tf

files = ['/tmp/wmt14/wpm/train.tfrecords-00000-of-00016', '/tmp/wmt14/wpm/train.tfrecords-00001-of-00016', ]
vocab_file = '/tmp/wmt14/wpm/wpm-ende.voc'
features = {
    'source_id': tf.io.VarLenFeature(tf.int64)
}
raw_dataset = tf.data.TFRecordDataset(files)
for raw_data in raw_dataset.shuffle(512).take(10):
    example = tf.train.Example()
    example.ParseFromString(raw_data.numpy())
    print(example)
    break

vocab = []

# \342\226\201: LOWER ONE EIGHTH BLOCK (_). 见https://www.unicodepedia.com/unicode/block-elements/2581/lower-one-eighth-block/
# 看上去是替换了所有的空格，见lingvo/tasks/mt/testdata/wmt14_ende_wpm_32k_test.vocab
# b'\342\226\201'.decode('utf-8')

with tf.io.gfile.GFile(vocab_file, 'r') as fr:
    for line in fr:
        ws = line.strip().split('\t')
        print(ws)
        vocab.append(ws[0].strip())
        if len(vocab) > 400:
            break
