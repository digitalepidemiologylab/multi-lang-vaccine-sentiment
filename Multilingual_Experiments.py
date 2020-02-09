################################
####### IMPORT MODULES #########
################################
import sys, os, json, csv, datetime, pprint, uuid, time, argparse

#Parse commandline
parser = argparse.ArgumentParser()
parser.add_argument("ip", help="IP-address of the TPU")
parser.add_argument(
    "-username",
    help=
    "Optional username of the one running the script. This is reflected in the directory name and in the logfile",
    default="Anonymous")
parser.add_argument("-iterations",
                    help="Number of times the script should run. Default is 1",
                    default=1,
                    type=int)
parser.add_argument(
    "-experiments",
    help=
    "Experiment number. Input as a string. You can use a syntax like \"2,3\" to run multiple experiments. Runs experiment #1 by default",
    default="1")
parser.add_argument("-epochs",
                    help="Number of train epochs. Default is 3",
                    default=3,
                    type=int)
parser.add_argument("-comment",
                    help="Add a Comment to the logfile",
                    default="No Comment")
args = parser.parse_args()

###################################
##### CLONING BERT AND THE DATA ###
###################################

#Clone Data
if not os.path.exists('data'):
    os.makedirs('data')
    os.system("gsutil -m cp -r gs://perepublic/EPFL_multilang/data/ .")
else:
    print('All training files has already been copied to data')

#Clone Bert
if not os.path.exists('bert_repo'):
    os.system(
        "test -d bert_repo || git clone https://github.com/google-research/bert bert_repo"
    )
else:
    print('The Bert repository has already been cloned')

if not '/content/bert_repo' in sys.path:
    sys.path += ['bert_repo']

###################################
#### IMPORT RTEMAINING LIBRARIES###
###################################
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
########## CONSTANTS #########
##############################
BERT_MODEL_DIR = 'gs://perepublic/multi_cased_L-12_H-768_A-12/'
BERT_MODEL_NAME = 'bert_model.ckpt'
BERT_MODEL_FILE = os.path.join(BERT_MODEL_DIR, BERT_MODEL_NAME)
TEMP_OUTPUT_BASEDIR = 'gs://perepublic/finetuned_models/'
TRAINING_LOG_FILE = '/home/per/multi-lang-vaccine-sentiment/trainlog.csv'

##############################
####### HYPERPARAMETERS#######
##############################
NUM_TRAIN_EPOCHS = args.epochs
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 128
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 8
PREDICT_BATCH_SIZE = 64
WARMUP_PROPORTION = 0.1

##############################
############ CONFIG ##########
##############################
SAVE_CHECKPOINTS_STEPS = 1000
SAVE_SUMMARY_STEPS = 500
USE_TPU = True
NUM_TPU_CORES = 8
ITERATIONS_PER_LOOP = 1000
LOWER_CASED = False


##############################
########### FUNCTIONS ########
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
        return ['positive', 'neutral', 'negative']

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
                run_classifier.InputExample(guid=guid,
                                            text_a=text_a,
                                            text_b=None,
                                            label=label))
        return examples



##############################
##### DEFINE EXPERIMENTS #####
##############################

experiment_definition = {
    "1": {
        "name":
        "zeroshot",
        "traindata": [
            'cb-annot-en', 'cb-annot-en', 'cb-annot-en', 'cb-annot-en',
            'cb-annot-en'
        ],
        "evaldata": [
            'cb-annot-en', 'cb-annot-en-de', 'cb-annot-en-es',
            'cb-annot-en-fr', 'cb-annot-en-pt'
        ]
    },
    "2": {
        "name":
        "zeroshot",
        "traindata": [
            'cb-annot-en', 'cb-annot-en-de', 'cb-annot-en-es',
            'cb-annot-en-fr', 'cb-annot-en-pt'
        ],
        "evaldata": [
            'cb-annot-en', 'cb-annot-en-de', 'cb-annot-en-es',
            'cb-annot-en-fr', 'cb-annot-en-pt'
        ]
    },
    "3": {
        "name":
        "zeroshot",
        "traindata": [
            'cb-annot-en', 'cb-annot-en', 'cb-annot-en', 'cb-annot-en',
            'cb-annot-en'
        ],
        "evaldata": [
            'cb-annot-en', 'cb-annot-en-de', 'cb-annot-en-es',
            'cb-annot-en-fr', 'cb-annot-en-pt'
        ]
    }
}



def run_experiment(experiments):
    #Interpret the input, and get all the experiments that should run into a list
    experiment_list = [x.strip() for x in experiments.split(',')]

    def get_run_config(output_dir):
        return tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=output_dir,
            save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=ITERATIONS_PER_LOOP,
                num_shards=NUM_TPU_CORES,
                per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.
                PER_HOST_V2))

    for exp_nr in experiment_list:
        print("***** Starting Experiment "+exp_nr+" *******")


if __name__ == "__main__":
    for i in range(0, args.iterations):
        run_experiment(args.experiments)
        print("*** Completed iteration " + str(i + 1))
