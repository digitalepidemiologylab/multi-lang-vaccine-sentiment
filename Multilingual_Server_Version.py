#@markdown ##Autenticate, Import Libraries and Copy Data Files
import sys, getopt
if len(sys.argv) !=3:
   print("Error. Provide an ip-address for the TPU and a unique username ")
   sys.exit()

print("IP set to: "+str(sys.argv[1]))


#import modules
import sys, os, json, csv, datetime, time, pprint
from google.colab import auth
from google.colab import drive
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.keras.utils import Progbar
import sklearn.metrics
USE_TPU = True


#Authenticate
auth.authenticate_user()

##Copy all the training data locally
if not os.path.exists('data'):
  os.makedirs('data')
  os.system("gsutil -m cp -r gs://perepublic/EPFL_multilang/data/ /home/per/content/")
else:
  print('All training files has already been copied to /home/per/content/data')

#Clone Bert
if not os.path.exists('bert_repo'):  
  os.system("test -d bert_repo || git clone https://github.com/google-research/bert bert_repo")
else:
  print('The Bert repository has already been cloned')

if not '/home/per/content/bert_repo' in sys.path:
  sys.path += ['bert_repo']
  sys.path += ['/home/per/content/bert_repo']

# import python modules defined by BERT
import modeling
import optimization
import run_classifier
import run_classifier_with_tfhub
import tokenization


TPU_IP = str(sys.argv[1])
TPU_ADDRESS = 'grpc://' + TPU_IP + ':8470'
print('Trying to allocate TPUs with address: ', TPU_ADDRESS)
with tf.Session(TPU_ADDRESS) as session:
  print('TPU devices:')
  pprint.pprint(session.list_devices())

#Define some custom functions
class vaccineStanceProcessor(run_classifier.DataProcessor):
  """Processor for the NoRec data set."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, 'train.tsv')), 'train')

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, 'dev.tsv')), 'dev')

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, 'test.tsv')), 'test')

  def get_labels(self):
    """See base class."""
    return ['positive','neutral','negative']

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      # Only the test set has a header
      if set_type == 'test' and i == 0:
        continue
      guid = '%s-%s' % (set_type, i)
      if set_type == 'test':
        text_a = tokenization.convert_to_unicode(line[3])
        #Set a dummy value. This is not used
        label = 'positive'
      else:
        text_a = tokenization.convert_to_unicode(line[3])
        label = tokenization.convert_to_unicode(line[1])
      examples.append(
          run_classifier.InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples

# Train the model
def model_train(estimator):

  # Time to train another model - Clean the output directory.
  #os.system("gsutil rm -r "+TEMP_OUTPUT_DIR)
  #vocabfile = os.path.join(BERT_MODEL_DIR,'vocab.txt')
  #configfile = os.path.join(BERT_MODEL_DIR,'bert_config.json')
  #os.system("gsutil cp "+vocabfile+" "+TEMP_OUTPUT_DIR+"/")
  #os.system("gsutil cp "+configfile+" "+TEMP_OUTPUT_DIR+"/")
  

  # Force TF Hub writes to the GS bucket we provide.
  os.environ['TFHUB_CACHE_DIR'] = TEMP_OUTPUT_DIR
  
  print('Fine tuning BERT base model normally takes a few minutes. Please wait...')
  # We'll set sequences to be at most 128 tokens long.
  train_features = run_classifier.convert_examples_to_features(
      train_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
  print('***** Started training at {} *****'.format(datetime.datetime.now()))
  print('  Num examples = {}'.format(len(train_examples)))
  print('  Batch size = {}'.format(TRAIN_BATCH_SIZE))
  print('  Train steps = {}'.format(num_train_steps))
  print('  Epochs = {}'.format(NUM_TRAIN_EPOCHS))
  
  tf.logging.info('  Num steps = %d', num_train_steps)
  train_input_fn = run_classifier.input_fn_builder(
      features=train_features,
      seq_length=MAX_SEQ_LENGTH,
      is_training=True,
      drop_remainder=True)
  estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
  print('***** Finished training at {} *****'.format(datetime.datetime.now()))

# Evaluate the model.
def model_eval(estimator):
  #Copy the vocab and config files from the base directory to the temporary finetune directory so that model can be restored 
  # Time to train another model - Clean the output directory.
  #vocabfile = os.path.join(BERT_MODEL_DIR,'vocab.txt')
  #configfile = os.path.join(BERT_MODEL_DIR,'bert_config.json')
  #os.system("gsutil cp "+vocabfile+" "+TEMP_OUTPUT_DIR+"/")
  #os.system("gsutil cp "+configfile+" "+TEMP_OUTPUT_DIR+"/")  
  #Just to make sure the files are really there for reading
  #time.sleep(3)


  eval_examples = processor.get_dev_examples(EVAL_ANNOT_DATASET_DIR)
  eval_features = run_classifier.convert_examples_to_features(
      eval_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
  print('***** Started evaluation at {} *****'.format(datetime.datetime.now()))
  print('Num examples = {}'.format(len(eval_examples)))
  print('Batch size = {}'.format(EVAL_BATCH_SIZE))

  # Eval will be slightly WRONG on the TPU because it will truncate
  # the last batch.
  eval_steps = int(len(eval_examples) / EVAL_BATCH_SIZE)
  eval_input_fn = run_classifier.input_fn_builder(
      features=eval_features,
      seq_length=MAX_SEQ_LENGTH,
      is_training=False,
      drop_remainder=True)
  result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
  print('***** Finished evaluation at {} *****'.format(datetime.datetime.now()))
  output_eval_file = os.path.join(TEMP_OUTPUT_DIR, 'eval_results.txt')
  with tf.gfile.GFile(output_eval_file, 'w') as writer:
    print('***** Eval results *****')
    for key in sorted(result.keys()):
      print('  {} = {}'.format(key, str(result[key])))
      writer.write('%s = %s\n' % (key, str(result[key])))
  predictions = estimator.predict(eval_input_fn)
  y_pred = [np.argmax(p['probabilities']) for p in predictions]
  y_true = [e.label_id for e in eval_features]
  label_mapping = dict(zip(range(len(label_list)), label_list))
  scores = performance_metrics(y_true, y_pred, label_mapping=label_mapping)
  print('Final scores:')
  print(scores)

  # Write log to Training Log File
  data = {'Experiment_Name': EXP_NAME,'Date': format(datetime.datetime.now()),'User': USERNAME, 'Model': BERT_MODEL_NAME, 'Train_Annot_Dataset': TRAIN_ANNOT_DATASET,'Eval_Annot_Dataset': EVAL_ANNOT_DATASET, 'Num_Train_Epochs': NUM_TRAIN_EPOCHS,'Learning_Rate': LEARNING_RATE, 'Max_Seq_Length': MAX_SEQ_LENGTH, 'Eval_Loss': result['eval_loss'],'Loss': result['loss'], 'Comment': COMMENT, **scores}
  datafields = sorted(data.keys())

  if not os.path.isfile(TRAINING_LOG_FILE):
    with open(TRAINING_LOG_FILE, mode='w') as output:
      output_writer = csv.DictWriter(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, fieldnames=datafields)
      output_writer.writeheader()
  with open(TRAINING_LOG_FILE, mode='a+') as output:
    output_writer = csv.DictWriter(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, fieldnames=datafields)
    output_writer.writerow(data)
    print("Wrote log to csv-file")

def performance_metrics(y_true, y_pred, metrics=None, averaging=None, label_mapping=None):
    """
    Compute performance metrics
    """
    def _compute_performance_metric(scoring_function, m, y_true, y_pred):
        for av in averaging:
            if av is None:
                metrics_by_class = scoring_function(y_true, y_pred, average=av, labels=labels)
                for i, class_metric in enumerate(metrics_by_class):
                    if label_mapping is None:
                        label_name = labels[i]
                    else:
                        label_name = label_mapping[labels[i]]
                    scores[m + '_' + str(label_name)] = class_metric
            else:
                scores[m + '_' + av] = scoring_function(y_true, y_pred, average=av, labels=labels)
    if averaging is None:
        averaging = ['micro', 'macro', 'weighted', None]
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1']
    scores = {}
    if label_mapping is None:
        # infer labels from data
        labels = sorted(list(set(y_true + y_pred)))
    else:
        labels = sorted(list(label_mapping.keys()))
    if len(labels) <= 2:
        # binary classification
        averaging += ['binary']
    for m in metrics:
        if m == 'accuracy':
            scores[m] = sklearn.metrics.accuracy_score(y_true, y_pred)
        elif m == 'precision':
            _compute_performance_metric(sklearn.metrics.precision_score, m, y_true, y_pred)
        elif m == 'recall':
            _compute_performance_metric(sklearn.metrics.recall_score, m, y_true, y_pred)
        elif m == 'f1':
            _compute_performance_metric(sklearn.metrics.f1_score, m, y_true, y_pred)
    return scores

def model_predict(estimator):
  # Make predictions on a subset of eval examples
  prediction_examples = processor.get_dev_examples(TASK_DATA_DIR)[:PREDICT_BATCH_SIZE]
  input_features = run_classifier.convert_examples_to_features(prediction_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
  predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=True)
  predictions = estimator.predict(predict_input_fn)

  for example, prediction in zip(prediction_examples, predictions):
    print('text_a: %s\nlabel:%s\nprediction:%s\n' % (example.text_a, str(example.label), prediction['probabilities']))

def model_init():
  #Initiation
  model_fn = run_classifier.model_fn_builder(
    bert_config=modeling.BertConfig.from_json_file(os.path.join(BERT_MODEL_DIR,'bert_config.json')),
    num_labels=len(label_list),
    init_checkpoint=BERT_MODEL_FILE,
    learning_rate=LEARNING_RATE,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps,
    use_tpu=USE_TPU,
    use_one_hot_embeddings=True
  )

  estimator_from_checkpoints = tf.contrib.tpu.TPUEstimator(
    use_tpu=USE_TPU,
    model_fn=model_fn,
    config=get_run_config(TEMP_OUTPUT_DIR),
    train_batch_size=TRAIN_BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE,
    predict_batch_size=PREDICT_BATCH_SIZE,
  )

  return estimator_from_checkpoints

def get_run_config(output_dir):
  return tf.contrib.tpu.RunConfig(
    cluster=tpu_cluster_resolver,
    model_dir=output_dir,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
    tpu_config=tf.contrib.tpu.TPUConfig(
        iterations_per_loop=ITERATIONS_PER_LOOP,
        num_shards=NUM_TPU_CORES,
        per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))




#@markdown ##Parameters
#@markdown Username and Comment are only used in temporal folders and in the train log file

#General Files and Settings
USERNAME = sys.argv[2] #@param {type:"string"}
#@markdown <br />
COMMENT = 'VM'#@param {type:"string"}
EXP_NAME = 'default-exp-name'#@param {type:"string"}
#@markdown <br />
BERT_MODEL_DIR = 'gs://perepublic/multi_cased_L-12_H-768_A-12/'#@param {type:"string"}
BERT_MODEL_NAME = 'bert_model.ckpt.index'#@param {type:"string"}
TEMP_OUTPUT_DIR = 'gs://perepublic/finetuned_models/' #@param {type:"string"}
TRAINING_LOG_FILE = '/home/per/multi-lang-log/trainlog.csv'#@param {type:"string"}

#@markdown <br />
TRAIN_ANNOT_DATASET = 'cb-annot-en' #@param ['cb-annot-en','cb-annot-en-de','cb-annot-en-es','cb-annot-en-fr','cb-annot-en-pt','cb-annot-en-sm','cb-annot-en-de-sm','cb-annot-en-es-sm','cb-annot-en-fr-sm','cb-annot-en-pt-sm']
EVAL_ANNOT_DATASET = 'cb-annot-en' #@param ['cb-annot-en','cb-annot-en-de','cb-annot-en-es','cb-annot-en-fr','cb-annot-en-pt','cb-annot-en-sm','cb-annot-en-de-sm','cb-annot-en-es-sm','cb-annot-en-fr-sm','cb-annot-en-pt-sm']


#Complete some paths
BERT_MODEL_FILE = os.path.join(BERT_MODEL_DIR,BERT_MODEL_NAME)
TRAIN_ANNOT_DATASET_DIR = os.path.join('/home/per/content/data',TRAIN_ANNOT_DATASET)
EVAL_ANNOT_DATASET_DIR = os.path.join('/home/per/content/data',EVAL_ANNOT_DATASET)


#Model Finetuning Hyperparameters
NUM_TRAIN_EPOCHS = 5
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 128
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 8
PREDICT_BATCH_SIZE = 64
WARMUP_PROPORTION = 0.1

#Other Config
SAVE_CHECKPOINTS_STEPS = 1000
SAVE_SUMMARY_STEPS = 500

NUM_TPU_CORES = 8
ITERATIONS_PER_LOOP = 1000
LOWER_CASED = False

#Do checks to see if all necessary files exists
if not tf.gfile.Exists(BERT_MODEL_DIR):
  print('Can not access the Bert model directory')
  sys.exit('Stopping execution!')

if not tf.gfile.Exists(TRAIN_ANNOT_DATASET_DIR):
  print('Can not access the training files')
  sys.exit('Stopping execution!')

if not tf.gfile.Exists(EVAL_ANNOT_DATASET_DIR):
  print('Can not access the training files')
  sys.exit('Stopping execution!')

if not tf.gfile.Exists(TEMP_OUTPUT_DIR):
  print('Can not access the temporary directory for storing finetuned models')
  sys.exit('Stopping execution!')
else:
  TEMP_OUTPUT_DIR += USERNAME

if not tf.gfile.Exists(os.path.join(BERT_MODEL_DIR,'vocab.txt')):
  print('Can not access the Bert model vocabulary file. This file should be located in the Bert model dir')
  sys.exit('Stopping execution!')

if not tf.gfile.Exists(os.path.join(BERT_MODEL_DIR,'bert_config.json')):
  print('Can not access the Bert model config file. This file should be located in the Bert model dir')
  sys.exit('Stopping execution!')

if not tf.gfile.Exists(BERT_MODEL_FILE):
  print('Can not access the Bert model file')
  sys.exit('Stopping execution!')

print ('All necessary files exist!')

tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(BERT_MODEL_DIR,'vocab.txt'),do_lower_case=LOWER_CASED)
tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(TPU_ADDRESS)
processor = vaccineStanceProcessor()
label_list = processor.get_labels()

# Compute number of train and warmup steps from batch size
train_examples = processor.get_train_examples(TRAIN_ANNOT_DATASET_DIR)
num_train_steps = int(len(train_examples) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)
print('There are a total of '+str(len(train_examples))+' training examples in '+TRAIN_ANNOT_DATASET_DIR+'.\nWe will be training for '+str(NUM_TRAIN_EPOCHS)+' epochs, which means '+str(num_train_steps)+' training steps with a batch size of '+str(TRAIN_BATCH_SIZE)+'.\n\n')



#Experiments
iterations = 10
import time

for _ in range(0,iterations):  
  #First experiment
  #zeroshot
  print("\nStarting first experiment - zeroshot")
  zeroshot_train = ['cb-annot-en']
  zeroshot_eval = ['cb-annot-en','cb-annot-en-de','cb-annot-en-es','cb-annot-en-fr','cb-annot-en-pt']

  for dataset in zeroshot_train:
    TRAIN_ANNOT_DATASET = dataset
    TRAIN_ANNOT_DATASET_DIR = os.path.join('/home/per/content/data', dataset)
    EXP_NAME = 'zeroshot-'+dataset
    print("Training " + EXP_NAME)
    estimator_from_checkpoints = model_init()
    model_train(estimator_from_checkpoints)
  
  for dataset in zeroshot_eval:
    EVAL_ANNOT_DATASET = dataset
    EVAL_ANNOT_DATASET_DIR = os.path.join('/home/per/content/data', dataset)
    EXP_NAME = 'zeroshot-'+dataset
    print("Evaluating " + EXP_NAME)
    model_eval(estimator_from_checkpoints)
  '''
  #Second experiment
  #translated
  print("\nStarting second experiment - translate")
  translated_train = ['cb-annot-en','cb-annot-en-de','cb-annot-en-es','cb-annot-en-fr','cb-annot-en-pt']
  translated_eval = ['cb-annot-en','cb-annot-en-de','cb-annot-en-es','cb-annot-en-fr','cb-annot-en-pt']

  for idx,dataset in enumerate(translated_train):
    TRAIN_ANNOT_DATASET = dataset
    TRAIN_ANNOT_DATASET_DIR = os.path.join('/home/per/content/data', dataset)
    EXP_NAME = 'translated-'+dataset
    print("Training " + EXP_NAME)
    estimator_from_checkpoints = model_init()
    model_train(estimator_from_checkpoints)

    EVAL_ANNOT_DATASET = translated_eval[idx]
    EVAL_ANNOT_DATASET_DIR = os.path.join('/home/per/content/data', translated_eval[idx])
    EXP_NAME = 'translated-'+translated_eval[idx]
    print("Evaluating " + EXP_NAME)
    model_eval(estimator_from_checkpoints)
    
  #Third experiment
  #zeroshot
  print("\nStarting third experiment - multitranslate")
  multitranslate_train = ['cb-annot-en-de-fr-es']
  multitranslate_eval = ['cb-annot-en','cb-annot-en-de','cb-annot-en-es','cb-annot-en-fr','cb-annot-en-pt']

  for dataset in multitranslate_train:
    TRAIN_ANNOT_DATASET = dataset
    TRAIN_ANNOT_DATASET_DIR = os.path.join('/home/per/content/data', dataset)
    EXP_NAME = 'multitranslate-'+dataset
    print("Training " + EXP_NAME)
    estimator_from_checkpoints = model_init()    
    model_train(estimator_from_checkpoints)

  for dataset in multitranslate_eval:
    EVAL_ANNOT_DATASET = dataset
    EVAL_ANNOT_DATASET_DIR = os.path.join('/home/per/content/data', dataset)
    EXP_NAME = 'multitranslate-'+dataset
    print("Evaluating " + EXP_NAME)
    model_eval(estimator_from_checkpoints)
  '''


