##############################
########import modules########
##############################
import sys, os, json, csv, datetime, pprint, uuid, time, argparse

#Parse commandline
parser = argparse.ArgumentParser()
parser.add_argument("ip", help="IP-address of the TPU")
parser.add_argument("-username", help="Optional username of the one running the script. This is reflected in the directory name and in the logfile", default="Anonymous")
parser.add_argument("-iterations", help="Number of times the script should run. Default is 1", default=1, type=int)
parser.add_argument("-experiments", help="Experiment number. Input as string. You can specify multiple experiments like \"1,3\". Runs experiment #1 by default",default="1")
parser.add_argument("-epochs", help="Number of train epochs. Default is 3", default=3, type=int)
parser.add_argument("-comment", help="Add a Comment to the logfile", default="No Comment")
args = parser.parse_args()

##############################
##Cloning Bert and the data###
##############################

#Clone Data
if not os.path.exists('data'):
  os.makedirs('data')
  os.system("gsutil -m cp -r gs://perepublic/EPFL_multilang/data/ .")
else:
  print('All training files has already been copied to data')

#Clone Bert
if not os.path.exists('bert_repo'):  
  os.system("test -d bert_repo || git clone https://github.com/google-research/bert bert_repo")
else:
  print('The Bert repository has already been cloned')

if not '/content/bert_repo' in sys.path:
  sys.path += ['bert_repo']

##############################
##Import remaining libraries##
##############################
from google.colab import auth
from google.colab import drive
from vac_utils import performance_metrics
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sklearn.metrics
import modeling
import optimization
import run_classifier
import run_classifier_with_tfhub
import tokenization



##############################
###########Constants##########
##############################
BERT_MODEL_DIR = 'gs://perepublic/multi_cased_L-12_H-768_A-12/'
BERT_MODEL_NAME = 'bert_model.ckpt'
BERT_MODEL_FILE = os.path.join(BERT_MODEL_DIR,BERT_MODEL_NAME)

TEMP_OUTPUT_BASEDIR = 'gs://perepublic/finetuned_models/'
TEMP_OUTPUT_DIR = os.path.join(TEMP_OUTPUT_BASEDIR, time.strftime('%Y-%m-%d%H:%M:%S') + "-"+str(args.username)+"-"+ str(uuid.uuid4()))

TRAINING_LOG_FILE = '/home/per/multi-lang-vaccine-sentiment/trainlog.csv'

##############################
######Hyper Parameters########
##############################
NUM_TRAIN_EPOCHS = args.epochs
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 128
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 8
PREDICT_BATCH_SIZE = 64
WARMUP_PROPORTION = 0.1

##############################
#############Config###########
##############################
SAVE_CHECKPOINTS_STEPS = 1000
SAVE_SUMMARY_STEPS = 500
USE_TPU = True
NUM_TPU_CORES = 8
ITERATIONS_PER_LOOP = 1000
LOWER_CASED = False


##############################
#############Functions########
##############################
def tpu_init():
    #Set up the TPU
    auth.authenticate_user()
    tpu_address = 'grpc://' + str(args.ip) + ':8470'
    
    print('TPU address is', tpu_address)
    with tf.Session(tpu_address) as session:
        print('TPU devices:')
        pprint.pprint(session.list_devices())
    
    return tpu_address


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



def run_experiment(experiments):
    experiment_list = [x.strip() for x in experiments.split(',')]
    
    def get_run_config(output_dir):
        return tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=output_dir,
            save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=ITERATIONS_PER_LOOP,
                num_shards=NUM_TPU_CORES,
                per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))

    ########################
    ##### EXPERIMENT 1 #####
    ########################
    if "1" in experiment_list:
        print("***** Starting Experiment 1 *******")
        zeroshot_train = ['cb-annot-en']
        zeroshot_eval = ['cb-annot-en','cb-annot-en-de','cb-annot-en-es','cb-annot-en-fr','cb-annot-en-pt']

        for train_annot_dataset in zeroshot_train:
            tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(BERT_MODEL_DIR,'vocab.txt'),do_lower_case=LOWER_CASED)
            tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(TPU_ADDRESS)
            processor = vaccineStanceProcessor()
            label_list = processor.get_labels()

            train_examples = processor.get_train_examples(os.path.join('data', train_annot_dataset))
            num_train_steps = int(len(train_examples) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
            num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)
        
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

            estimator = tf.contrib.tpu.TPUEstimator(
                use_tpu=USE_TPU,
                model_fn=model_fn,
                config=get_run_config(TEMP_OUTPUT_DIR),
                train_batch_size=TRAIN_BATCH_SIZE,
                eval_batch_size=EVAL_BATCH_SIZE,
                predict_batch_size=PREDICT_BATCH_SIZE,
            )

            os.environ['TFHUB_CACHE_DIR'] = TEMP_OUTPUT_DIR

            train_features = run_classifier.convert_examples_to_features(
                train_examples, label_list, MAX_SEQ_LENGTH, tokenizer)

            print('Fine tuning BERT base model normally takes a few minutes. Please wait...')            
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

        for eval_annot_dataset in zeroshot_eval:
            EXP_NAME = 'zeroshot-(train)-'+train_annot_dataset+"-(eval)-"+eval_annot_dataset
            
            eval_examples = processor.get_dev_examples(os.path.join('data', eval_annot_dataset))
            eval_features = run_classifier.convert_examples_to_features(
                eval_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
            print('***** Started evaluation of {} at {} *****'.format(EXP_NAME, datetime.datetime.now()))
            print('Num examples = {}'.format(len(eval_examples)))
            print('Batch size = {}'.format(EVAL_BATCH_SIZE))

            # Eval will be slightly WRONG on the TPU because it will truncate the last batch. 
            eval_steps = int(len(eval_examples) / EVAL_BATCH_SIZE)
            eval_input_fn = run_classifier.input_fn_builder(
                features=eval_features,
                seq_length=MAX_SEQ_LENGTH,
                is_training=False,
                drop_remainder=True)
            result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

            print('***** Finished first half of evaluation at {} *****'.format(datetime.datetime.now()))
            
            output_eval_file = os.path.join(TEMP_OUTPUT_DIR, 'eval_results.txt')
            with tf.gfile.GFile(output_eval_file, 'w') as writer:
                print('***** Eval results *****')
                for key in sorted(result.keys()):
                    print(f" {key} = {result[key]}")
                    writer.write('%s = %s\n' % (key, f"{result[key]}"))

            predictions = estimator.predict(eval_input_fn)
            y_pred = [np.argmax(p['probabilities']) for p in predictions]
            y_true = [e.label_id for e in eval_features]
            label_mapping = dict(zip(range(len(label_list)), label_list))
            scores = performance_metrics(y_true, y_pred, label_mapping=label_mapping)
            print('Final scores:')
            print(scores)
            print('***** Finished second half of evaluation at {} *****'.format(datetime.datetime.now()))


            # Write log to Training Log File
            data = {'Experiment_Name': EXP_NAME,'Date': format(datetime.datetime.now()),'User': args.username, 'Model': BERT_MODEL_NAME, 'Train_Annot_Dataset': train_annot_dataset,'Eval_Annot_Dataset': eval_annot_dataset, 'Num_Train_Epochs': NUM_TRAIN_EPOCHS,'Learning_Rate': LEARNING_RATE, 'Max_Seq_Length': MAX_SEQ_LENGTH, 'Eval_Loss': result['eval_loss'],'Loss': result['loss'], 'Comment': args.comment, **scores}
            datafields = sorted(data.keys())

            if not os.path.isfile(TRAINING_LOG_FILE):
                with open(TRAINING_LOG_FILE, mode='w') as output:
                    output_writer = csv.DictWriter(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, fieldnames=datafields)
                    output_writer.writeheader()
            with open(TRAINING_LOG_FILE, mode='a+') as output:
                output_writer = csv.DictWriter(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, fieldnames=datafields)
                output_writer.writerow(data)
                print("Wrote log to csv-file")
        print("***** Completed Experiment 1 *******")

    ########################
    ##### EXPERIMENT 2 #####
    ########################
    if "2" in experiment_list:
        print("***** Starting Experiment 2 *******")
        translated_train = ['cb-annot-en','cb-annot-en-de','cb-annot-en-es','cb-annot-en-fr','cb-annot-en-pt']
        translated_eval = ['cb-annot-en','cb-annot-en-de','cb-annot-en-es','cb-annot-en-fr','cb-annot-en-pt']
        
        for idx,train_annot_dataset in enumerate(translated_train):
            eval_annot_dataset = train_annot_dataset[idx]

            tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(BERT_MODEL_DIR,'vocab.txt'),do_lower_case=LOWER_CASED)
            tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(TPU_ADDRESS)
            processor = vaccineStanceProcessor()
            label_list = processor.get_labels()

            train_examples = processor.get_train_examples(os.path.join('data', train_annot_dataset))
            num_train_steps = int(len(train_examples) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
            num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)
        
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

            estimator = tf.contrib.tpu.TPUEstimator(
                use_tpu=USE_TPU,
                model_fn=model_fn,
                config=get_run_config(TEMP_OUTPUT_DIR),
                train_batch_size=TRAIN_BATCH_SIZE,
                eval_batch_size=EVAL_BATCH_SIZE,
                predict_batch_size=PREDICT_BATCH_SIZE,
            )

            os.environ['TFHUB_CACHE_DIR'] = TEMP_OUTPUT_DIR

            train_features = run_classifier.convert_examples_to_features(
                train_examples, label_list, MAX_SEQ_LENGTH, tokenizer)

            print('Fine tuning BERT base model normally takes a few minutes. Please wait...')            
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

            EXP_NAME = 'translated-(train)-'+train_annot_dataset+"-(eval)-"+eval_annot_dataset
            
            eval_examples = processor.get_dev_examples(os.path.join('data', eval_annot_dataset))
            eval_features = run_classifier.convert_examples_to_features(
                eval_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
            print('***** Started evaluation of {} at {} *****'.format(EXP_NAME, datetime.datetime.now()))
            print('Num examples = {}'.format(len(eval_examples)))
            print('Batch size = {}'.format(EVAL_BATCH_SIZE))

            # Eval will be slightly WRONG on the TPU because it will truncate the last batch. 
            eval_steps = int(len(eval_examples) / EVAL_BATCH_SIZE)
            eval_input_fn = run_classifier.input_fn_builder(
                features=eval_features,
                seq_length=MAX_SEQ_LENGTH,
                is_training=False,
                drop_remainder=True)
            result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

            print('***** Finished first half of evaluation at {} *****'.format(datetime.datetime.now()))
            
            output_eval_file = os.path.join(TEMP_OUTPUT_DIR, 'eval_results.txt')
            with tf.gfile.GFile(output_eval_file, 'w') as writer:
                print('***** Eval results *****')
                for key in sorted(result.keys()):
                    print(f" {key} = {result[key]}")
                    writer.write('%s = %s\n' % (key, f"{result[key]}"))
            predictions = estimator.predict(eval_input_fn)
            y_pred = [np.argmax(p['probabilities']) for p in predictions]
            y_true = [e.label_id for e in eval_features]
            label_mapping = dict(zip(range(len(label_list)), label_list))
            scores = performance_metrics(y_true, y_pred, label_mapping=label_mapping)
            print('Final scores:')
            print(scores)
            print('***** Finished second half of evaluation at {} *****'.format(datetime.datetime.now()))


            # Write log to Training Log File
            data = {'Experiment_Name': EXP_NAME,'Date': format(datetime.datetime.now()),'User': args.username, 'Model': BERT_MODEL_NAME, 'Train_Annot_Dataset': train_annot_dataset,'Eval_Annot_Dataset': eval_annot_dataset, 'Num_Train_Epochs': NUM_TRAIN_EPOCHS,'Learning_Rate': LEARNING_RATE, 'Max_Seq_Length': MAX_SEQ_LENGTH, 'Eval_Loss': result['eval_loss'],'Loss': result['loss'], 'Comment': args.comment, **scores}
            datafields = sorted(data.keys())

            if not os.path.isfile(TRAINING_LOG_FILE):
                with open(TRAINING_LOG_FILE, mode='w') as output:
                    output_writer = csv.DictWriter(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, fieldnames=datafields)
                    output_writer.writeheader()
            with open(TRAINING_LOG_FILE, mode='a+') as output:
                output_writer = csv.DictWriter(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, fieldnames=datafields)
                output_writer.writerow(data)
                print("Wrote log to csv-file")
        print("***** Completed Experiment 2 *******")

    ########################
    ##### EXPERIMENT 3 #####
    ########################
    if "1" in experiment_list:
        print("***** Starting Experiment 3 *******")
        multitranslate_train = ['cb-annot-en-de-fr-es']
        multitranslate_eval = ['cb-annot-en','cb-annot-en-de','cb-annot-en-es','cb-annot-en-fr','cb-annot-en-pt']

        for train_annot_dataset in multitranslate_train:
            tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(BERT_MODEL_DIR,'vocab.txt'),do_lower_case=LOWER_CASED)
            tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(TPU_ADDRESS)
            processor = vaccineStanceProcessor()
            label_list = processor.get_labels()

            train_examples = processor.get_train_examples(os.path.join('data', train_annot_dataset))
            num_train_steps = int(len(train_examples) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
            num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)
        
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

            estimator = tf.contrib.tpu.TPUEstimator(
                use_tpu=USE_TPU,
                model_fn=model_fn,
                config=get_run_config(TEMP_OUTPUT_DIR),
                train_batch_size=TRAIN_BATCH_SIZE,
                eval_batch_size=EVAL_BATCH_SIZE,
                predict_batch_size=PREDICT_BATCH_SIZE,
            )

            os.environ['TFHUB_CACHE_DIR'] = TEMP_OUTPUT_DIR

            train_features = run_classifier.convert_examples_to_features(
                train_examples, label_list, MAX_SEQ_LENGTH, tokenizer)

            print('Fine tuning BERT base model normally takes a few minutes. Please wait...')            
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

        for eval_annot_dataset in multitranslate_eval:
            EXP_NAME = 'multitranslate-(train)-'+train_annot_dataset+"-(eval)-"+eval_annot_dataset
            
            eval_examples = processor.get_dev_examples(os.path.join('data', eval_annot_dataset))
            eval_features = run_classifier.convert_examples_to_features(
                eval_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
            print('***** Started evaluation of {} at {} *****'.format(EXP_NAME, datetime.datetime.now()))
            print('Num examples = {}'.format(len(eval_examples)))
            print('Batch size = {}'.format(EVAL_BATCH_SIZE))

            # Eval will be slightly WRONG on the TPU because it will truncate the last batch. 
            eval_steps = int(len(eval_examples) / EVAL_BATCH_SIZE)
            eval_input_fn = run_classifier.input_fn_builder(
                features=eval_features,
                seq_length=MAX_SEQ_LENGTH,
                is_training=False,
                drop_remainder=True)
            result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

            print('***** Finished first half of evaluation at {} *****'.format(datetime.datetime.now()))
            
            output_eval_file = os.path.join(TEMP_OUTPUT_DIR, 'eval_results.txt')
            with tf.gfile.GFile(output_eval_file, 'w') as writer:
                print('***** Eval results *****')
                for key in sorted(result.keys()):
                    print(f" {key} = {result[key]}")
                    writer.write('%s = %s\n' % (key, f"{result[key]}"))
            predictions = estimator.predict(eval_input_fn)
            y_pred = [np.argmax(p['probabilities']) for p in predictions]
            y_true = [e.label_id for e in eval_features]
            label_mapping = dict(zip(range(len(label_list)), label_list))
            scores = performance_metrics(y_true, y_pred, label_mapping=label_mapping)
            print('Final scores:')
            print(scores)
            print('***** Finished second half of evaluation at {} *****'.format(datetime.datetime.now()))


            # Write log to Training Log File
            data = {'Experiment_Name': EXP_NAME,'Date': format(datetime.datetime.now()),'User': args.username, 'Model': BERT_MODEL_NAME, 'Train_Annot_Dataset': train_annot_dataset,'Eval_Annot_Dataset': eval_annot_dataset, 'Num_Train_Epochs': NUM_TRAIN_EPOCHS,'Learning_Rate': LEARNING_RATE, 'Max_Seq_Length': MAX_SEQ_LENGTH, 'Eval_Loss': result['eval_loss'],'Loss': result['loss'], 'Comment': args.comment, **scores}
            datafields = sorted(data.keys())

            if not os.path.isfile(TRAINING_LOG_FILE):
                with open(TRAINING_LOG_FILE, mode='w') as output:
                    output_writer = csv.DictWriter(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, fieldnames=datafields)
                    output_writer.writeheader()
            with open(TRAINING_LOG_FILE, mode='a+') as output:
                output_writer = csv.DictWriter(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, fieldnames=datafields)
                output_writer.writerow(data)
                print("Wrote log to csv-file")
        print("***** Completed Experiment 3 *******")



    print ("****Finished running all experiments!")


if __name__== "__main__":
    TPU_ADDRESS = tpu_init()

    for i in range(0,args.iterations):
        run_experiment(args.experiments)
        print(f"*** Completed iteration {i + 1}")


set