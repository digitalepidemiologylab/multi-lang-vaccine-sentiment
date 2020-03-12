################################
####### IMPORT MODULES #########
################################
import sys, os, json, csv, datetime, pprint, uuid, time, argparse, logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.get_logger(__name__)

###################################
##### CLONING BERT AND THE DATA ###
###################################

# Clone Data
if not os.path.exists('data'):
    os.makedirs('data')
    os.system("gsutil -m cp -r gs://perepublic/EPFL_multilang/data/ .")
else:
    logger.info('** All training files has already been copied to data')

# Clone Bert
if not os.path.exists('bert_repo'):
    os.system(
        "test -d bert_repo || git clone https://github.com/google-research/bert bert_repo"
    )
else:
    logger.info('** The Bert repository has already been cloned')

if not 'bert_repo' in sys.path:
    sys.path += ['bert_repo']

###################################
##### IMPORT REMAINING MODULES ####
###################################
from google.colab import auth
from google.colab import drive
from vac_utils import performance_metrics, get_predictions_output, append_to_csv, save_to_json
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
LOG_CSV_DIR = 'log_csv/'
PREDICTIONS_JSON_DIR = 'predictions_json/'
HIDDEN_STATE_JSON_DIR = 'hidden_state_json/'

logdirs = [LOG_CSV_DIR, PREDICTIONS_JSON_DIR, HIDDEN_STATE_JSON_DIR]

for d in logdirs:
    if not os.path.exists(d):
        os.makedirs(d)

##############################
####### HYPERPARAMETERS ######
##############################
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

NUM_TPU_CORES = 8
ITERATIONS_PER_LOOP = 1000
LOWER_CASED = False

##############################
########### FUNCTIONS ########
##############################
def tpu_init(ip):
    #Set up the TPU
    auth.authenticate_user()
    tpu_address = 'grpc://' + str(ip) + ':8470'

    with tf.Session(tpu_address) as session:
        logger.info('TPU devices:')
        pprint.pprint(session.list_devices())
    logger.info(f'TPU address is active on {tpu_address}')
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
            #guid = '%s-%s' % (set_type, i)
            guid = tokenization.convert_to_unicode(line[0])
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

experiment_definitions = {
    "1": {
        "name": "zeroshot-cb-annot-en-cb-annot-en",
        "train_annot_dataset": "cb-annot-en",
        "eval_annot_dataset": "cb-annot-en"
    },
    "2": {
        "name": "zeroshot-cb-annot-en-cb-annot-de",
        "train_annot_dataset": "cb-annot-en",
        "eval_annot_dataset": "cb-annot-en-de"
    },
    "3": {
        "name": "zeroshot-cb-annot-en-cb-annot-es",
        "train_annot_dataset": "cb-annot-en",
        "eval_annot_dataset": "cb-annot-en-es"
    },
    "4": {
        "name": "zeroshot-cb-annot-en-cb-annot-fr",
        "train_annot_dataset": "cb-annot-en",
        "eval_annot_dataset": "cb-annot-en-fr"
    },
    "5": {
        "name": "zeroshot-cb-annot-en-cb-annot-pt",
        "train_annot_dataset": "cb-annot-en",
        "eval_annot_dataset": "cb-annot-en-pt"
    },
    "6": {
        "name": "translate-cb-annot-en-cb-annot-en",
        "train_annot_dataset": "cb-annot-en",
        "eval_annot_dataset": "cb-annot-en"
    },
    "7": {
        "name": "translate-cb-annot-de-cb-annot-de",
        "train_annot_dataset": "cb-annot-en-de",
        "eval_annot_dataset": "cb-annot-en-de"
    },
    "8": {
        "name": "translate-cb-annot-es-cb-annot-es",
        "train_annot_dataset": "cb-annot-en-es",
        "eval_annot_dataset": "cb-annot-en-es"
    },
    "9": {
        "name": "translate-cb-annot-fr-cb-annot-fr",
        "train_annot_dataset": "cb-annot-en-fr",
        "eval_annot_dataset": "cb-annot-en-fr"
    },
    "10": {
        "name": "translate-cb-annot-pt-cb-annot-pt",
        "train_annot_dataset": "cb-annot-en-pt",
        "eval_annot_dataset": "cb-annot-en-pt"
    },
    "11": {
        "name": "multitranslate-cb-annot-en-de-fr-es-pt-cb-annot-en",
        "train_annot_dataset": "cb-annot-en-de-fr-es-pt",
        "eval_annot_dataset": "cb-annot-en"
    },
    "12": {
        "name": "multitranslate-cb-annot-en-de-fr-es-pt-cb-annot-de",
        "train_annot_dataset": "cb-annot-en-de-fr-es-pt",
        "eval_annot_dataset": "cb-annot-en-de"
    },
    "13": {
        "name": "multitranslate-cb-annot-en-de-fr-es-pt-cb-annot-es",
        "train_annot_dataset": "cb-annot-en-de-fr-es-pt",
        "eval_annot_dataset": "cb-annot-en-es"
    },
    "14": {
        "name": "multitranslate-cb-annot-en-de-fr-es-pt-cb-annot-fr",
        "train_annot_dataset": "cb-annot-en-de-fr-es-pt",
        "eval_annot_dataset": "cb-annot-en-fr"
    },
    "15": {
        "name": "multitranslate-cb-annot-en-de-fr-es-pt-cb-annot-pt",
        "train_annot_dataset": "cb-annot-en-de-fr-es-pt",
        "eval_annot_dataset": "cb-annot-en-pt"
    },
    "16": {
        "name": "zeroshot-small-cb-annot-en-sm-cb-annot-en",
        "train_annot_dataset": "cb-annot-en-sm",
        "eval_annot_dataset": "cb-annot-en"
    },
    "17": {
        "name": "zeroshot-small-cb-annot-en-sm-cb-annot-de",
        "train_annot_dataset": "cb-annot-en-sm",
        "eval_annot_dataset": "cb-annot-en-de"
    },
    "18": {
        "name": "zeroshot-small-cb-annot-en-sm-cb-annot-es",
        "train_annot_dataset": "cb-annot-en-sm",
        "eval_annot_dataset": "cb-annot-en-es"
    },
    "19": {
        "name": "zeroshot-small-cb-annot-en-sm-cb-annot-fr",
        "train_annot_dataset": "cb-annot-en-sm",
        "eval_annot_dataset": "cb-annot-en-fr"
    },
    "20": {
        "name": "zeroshot-small-cb-annot-en-sm-cb-annot-pt",
        "train_annot_dataset": "cb-annot-en-sm",
        "eval_annot_dataset": "cb-annot-en-pt"
    },
    "21": {
        "name": "translate-small-cb-annot-en-sm-cb-annot-en",
        "train_annot_dataset": "cb-annot-en-sm",
        "eval_annot_dataset": "cb-annot-en"
    },
    "22": {
        "name": "translate-small-cb-annot-de-sm-cb-annot-de",
        "train_annot_dataset": "cb-annot-en-de-sm",
        "eval_annot_dataset": "cb-annot-en-de"
    },
    "23": {
        "name": "translate-small-cb-annot-es-sm-cb-annot-es",
        "train_annot_dataset": "cb-annot-en-es-sm",
        "eval_annot_dataset": "cb-annot-en-es"
    },
    "24": {
        "name": "translate-small-cb-annot-fr-sm-cb-annot-fr",
        "train_annot_dataset": "cb-annot-en-fr-sm",
        "eval_annot_dataset": "cb-annot-en-fr"
    },
    "25": {
        "name": "translate-small-cb-annot-pt-sm-cb-annot-pt",
        "train_annot_dataset": "cb-annot-en-pt-sm",
        "eval_annot_dataset": "cb-annot-en-pt"
    },
    "26": {
        "name": "multitranslate-small-cb-annot-en-de-fr-es-pt-sm-cb-annot-en",
        "train_annot_dataset": "cb-annot-en-de-fr-es-pt-sm",
        "eval_annot_dataset": "cb-annot-en"
    },
    "27": {
        "name": "multitranslate-small-cb-annot-en-de-fr-es-pt-sm-cb-annot-de",
        "train_annot_dataset": "cb-annot-en-de-fr-es-pt-sm",
        "eval_annot_dataset": "cb-annot-en-de"
    },
    "28": {
        "name": "multitranslate-small-cb-annot-en-de-fr-es-pt-sm-cb-annot-es",
        "train_annot_dataset": "cb-annot-en-de-fr-es-pt-sm",
        "eval_annot_dataset": "cb-annot-en-es"
    },
    "29": {
        "name": "multitranslate-small-cb-annot-en-de-fr-es-pt-sm-cb-annot-fr",
        "train_annot_dataset": "cb-annot-en-de-fr-es-pt-sm",
        "eval_annot_dataset": "cb-annot-en-fr"
    },
    "30": {
        "name": "multitranslate-small-cb-annot-en-de-fr-es-pt-sm-cb-annot-pt",
        "train_annot_dataset": "cb-annot-en-de-fr-es-pt-sm",
        "eval_annot_dataset": "cb-annot-en-pt"
    },
    "31": {
        "name": "balanced-cb-annot-en-de-fr-es-pt-os-cb-annot-en",
        "train_annot_dataset": "cb-annot-en-de-fr-es-pt-os",
        "eval_annot_dataset": "cb-annot-en"
    },
    "32": {
        "name": "balanced-cb-annot-en-de-fr-es-pt-os-cb-annot-de",
        "train_annot_dataset": "cb-annot-en-de-fr-es-pt-os",
        "eval_annot_dataset": "cb-annot-en-de"
    },
    "33": {
        "name": "balanced-cb-annot-en-de-fr-es-pt-os-cb-annot-es",
        "train_annot_dataset": "cb-annot-en-de-fr-es-pt-os",
        "eval_annot_dataset": "cb-annot-en-es"
    },
    "34": {
        "name": "balanced-cb-annot-en-de-fr-es-pt-os-cb-annot-fr",
        "train_annot_dataset": "cb-annot-en-de-fr-es-pt-os",
        "eval_annot_dataset": "cb-annot-en-fr"
    },
    "35": {
        "name": "balanced-cb-annot-en-de-fr-es-pt-os-cb-annot-pt",
        "train_annot_dataset": "cb-annot-en-de-fr-es-pt-os",
        "eval_annot_dataset": "cb-annot-en-pt"
    },
    "36": {
        "name": "balanced-cb-annot-en-de-fr-es-pt-us-cb-annot-en",
        "train_annot_dataset": "cb-annot-en-de-fr-es-pt-us",
        "eval_annot_dataset": "cb-annot-en"
    },
    "37": {
        "name": "balanced-cb-annot-en-de-fr-es-pt-us-cb-annot-de",
        "train_annot_dataset": "cb-annot-en-de-fr-es-pt-us",
        "eval_annot_dataset": "cb-annot-en-de"
    },
    "38": {
        "name": "balanced-cb-annot-en-de-fr-es-pt-us-cb-annot-es",
        "train_annot_dataset": "cb-annot-en-de-fr-es-pt-us",
        "eval_annot_dataset": "cb-annot-en-es"
    },
    "39": {
        "name": "balanced-cb-annot-en-de-fr-es-pt-us-cb-annot-fr",
        "train_annot_dataset": "cb-annot-en-de-fr-es-pt-us",
        "eval_annot_dataset": "cb-annot-en-fr"
    },
    "40": {
        "name": "balanced-cb-annot-en-de-fr-es-pt-us-cb-annot-pt",
        "train_annot_dataset": "cb-annot-en-de-fr-es-pt-us",
        "eval_annot_dataset": "cb-annot-en-pt"
    }
}

###########################
##### RUN EXPERIMENTS #####
###########################


def run_experiment(experiments, use_tpu, tpu_address, repeat, num_train_steps, username,
                   comment):
    #Interpret the input, and get all the experiments that should run into a list
    experiment_list = [x.strip() for x in experiments.split(',')]


    logger.info(f'Getting ready to run the following experiments for {repeat} repeats: {experiment_list}')

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

    last_completed_train = ""
    completed_train_dirs = []

    for exp_nr in experiment_list:
        logger.info(f"***** Starting Experiment {exp_nr} *******")
        logger.info(f"***** {experiment_definitions[exp_nr]['name']} ******")
        logger.info("***********************************************")

        #Get a unique ID for every experiment run
        experiment_id = str(uuid.uuid4())

        ###########################
        ######### TRAINING ########
        ###########################

        #We should only train a new model if a similar model hasnt just been trained. Save considerable computation time
        train_annot_dataset = experiment_definitions[exp_nr][
            "train_annot_dataset"]


        if train_annot_dataset != last_completed_train:
            #Set a fresh new output directory every time training starts, and set the cache to this directory
            temp_output_dir = os.path.join(
                TEMP_OUTPUT_BASEDIR,experiment_id)

            os.environ['TFHUB_CACHE_DIR'] = temp_output_dir
            logger.info(f"***** Setting temporary dir {temp_output_dir} **")
            logger.info(f"***** Train started in {temp_output_dir} **")

            tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(
                BERT_MODEL_DIR, 'vocab.txt'),
                                                   do_lower_case=LOWER_CASED)


            if tpu_address:
                tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_address)
            else:
                tpu_cluster_resolver = None

            processor = vaccineStanceProcessor()
            label_list = processor.get_labels()
            label_mapping = dict(zip(range(len(label_list)), label_list))

            train_examples = processor.get_train_examples(
                os.path.join('data', train_annot_dataset))
            num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

            #Initiation
            model_fn = run_classifier.model_fn_builder(
                bert_config=modeling.BertConfig.from_json_file(
                    os.path.join(BERT_MODEL_DIR, 'bert_config.json')),
                num_labels=len(label_list),
                init_checkpoint=BERT_MODEL_FILE,
                learning_rate=LEARNING_RATE,
                num_train_steps=num_train_steps,
                num_warmup_steps=num_warmup_steps,
                use_tpu=use_tpu,
                use_one_hot_embeddings=True)

            estimator = tf.contrib.tpu.TPUEstimator(
                use_tpu=use_tpu,
                model_fn=model_fn,
                config=get_run_config(temp_output_dir),
                train_batch_size=TRAIN_BATCH_SIZE,
                eval_batch_size=EVAL_BATCH_SIZE,
                predict_batch_size=PREDICT_BATCH_SIZE,
            )

            train_features = run_classifier.convert_examples_to_features(
                train_examples, label_list, MAX_SEQ_LENGTH, tokenizer)

            logger.info('***** Fine tuning BERT base model normally takes a few minutes. Please wait...')
            logger.info('***** Started training using {} at {} *****'.format(train_annot_dataset, datetime.datetime.now()))
            logger.info('  Num examples = {}'.format(len(train_examples)))
            logger.info('  Batch size = {}'.format(TRAIN_BATCH_SIZE))
            logger.info('  Train steps = {}'.format(num_train_steps))
            logger.info('  Number of training steps = {}'.format(num_train_steps))

            tf.logging.info('  Num steps = %d', num_train_steps)
            train_input_fn = run_classifier.input_fn_builder(
                features=train_features,
                seq_length=MAX_SEQ_LENGTH,
                is_training=True,
                drop_remainder=True)

            estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
            logger.info('***** Finished training using {} at {} *****'.format(train_annot_dataset, datetime.datetime.now()))

            last_completed_train = train_annot_dataset
            completed_train_dirs.append(temp_output_dir)

            ######################################
            ######### TRAINING PREDICTION ########
            ######################################
            train_pred_input_fn = run_classifier.input_fn_builder(
                features=train_features,
                seq_length=MAX_SEQ_LENGTH,
                is_training=False,
                drop_remainder=False)

            predictions = estimator.predict(input_fn=train_pred_input_fn)
            probabilities = np.array([p['probabilities'] for p in predictions])
            y_true = [e.label_id for e in train_features]

            guid = [e.guid for e in train_examples]

            predictions_output = get_predictions_output(experiment_id, guid, probabilities, y_true, label_mapping=label_mapping)
            import pdb; pdb.set_trace()

            save_to_json(predictions_output ,os.path.join(PREDICTIONS_JSON_DIR,'train_'+experiment_id+'.json'))


        #############################
        ######### EVALUATING ########
        #############################
        eval_annot_dataset = experiment_definitions[exp_nr][
            "eval_annot_dataset"]

        eval_examples = processor.get_dev_examples(
            os.path.join('data', eval_annot_dataset))
        eval_features = run_classifier.convert_examples_to_features(
            eval_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
        logger.info('***** Started evaluation of {} at {} *****'.format(
            experiment_definitions[exp_nr]["name"], datetime.datetime.now()))
        logger.info('Num examples = {}'.format(len(eval_examples)))
        logger.info('Batch size = {}'.format(EVAL_BATCH_SIZE))

        # Eval will be slightly WRONG on the TPU because it will truncate the last batch.
        eval_steps = int(len(eval_examples) / EVAL_BATCH_SIZE)
        eval_input_fn = run_classifier.input_fn_builder(
            features=eval_features,
            seq_length=MAX_SEQ_LENGTH,
            is_training=False,
            drop_remainder=True)
        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        logger.info(
            '***** Finished first half of evaluation of {} at {} *****'.format(
                experiment_definitions[exp_nr]["name"],
                datetime.datetime.now()))

        output_eval_file = os.path.join(temp_output_dir, 'eval_results.txt')
        with tf.gfile.GFile(output_eval_file, 'w') as writer:
            logger.info('***** Eval results *****')
            for key in sorted(result.keys()):
                logger.info('  {} = {}'.format(key, str(result[key])))
                writer.write('%s = %s\n' % (key, str(result[key])))

        predictions = estimator.predict(eval_input_fn)
        probabilities = np.array([p['probabilities'] for p in predictions])
        y_pred = np.argmax(probabilities, axis=1)
        y_true = [e.label_id for e in eval_features]
        guid = [e.guid for e in eval_examples]
        scores = performance_metrics(y_true,
                                     y_pred,
                                     label_mapping=label_mapping)
        logger.info('Final scores:')
        logger.info(scores)
        logger.info('***** Finished second half of evaluation of {} at {} *****'.
              format(experiment_definitions[exp_nr]["name"],
                     datetime.datetime.now()))

        # write full dev prediction output
        predictions_output = get_predictions_output(experiment_id, guid, probabilities, y_true, label_mapping=label_mapping)
        save_to_json(predictions_output, os.path.join(PREDICTIONS_JSON_DIR,'dev_'+experiment_id+'.json'))

        # Write log to Training Log File
        data = {
            'Experiment_Name': experiment_definitions[exp_nr]["name"],
            'Experiment_Id':experiment_id,
            'Date': format(datetime.datetime.now()),
            'User': username,
            'Model': BERT_MODEL_NAME,
            'Num_Train_Steps': num_train_steps,
            'Train_Annot_Dataset': train_annot_dataset,
            'Eval_Annot_Dataset': eval_annot_dataset,
            'Learning_Rate': LEARNING_RATE,
            'Max_Seq_Length': MAX_SEQ_LENGTH,
            'Eval_Loss': result['eval_loss'],
            'Loss': result['loss'],
            'Comment': comment,
            **scores
        }

        append_to_csv(data, os.path.join(LOG_CSV_DIR,'fulltrainlog.csv'))
        logger.info(f"***** Completed Experiment {exp_nr} *******")

    logger.info(f"***** Completed all experiments in {repeat} repeats. We should now clean up all remaining files *****")
    for c in completed_train_dirs:
        logger.info("Deleting these directories: ")
        logger.info("gsutil -m rm -r " + c)
        os.system("gsutil -m rm -r " + c)

def parse_args(args):
    # Parse commandline
    parser = argparse.ArgumentParser()
    parser.add_argument("-ip",
                        "--tpu_ip",
                        dest='tpu_ip',
                        default=None,
                        help="IP-address of the TPU")
    parser.add_argument("-tpu",
                        "--use_tpu",
                        dest='use_tpu',
                        default=1,
                        help="Use TPU. Set to 1 or 0. If set to false, GPU will be used instead")
    parser.add_argument(
        "-u",
        "--username",
        help=
        "Optional. Username is used in the directory name and in the logfile",
        default="Anonymous")
    parser.add_argument(
        "-r",
        "--repeats",
        help="Number of times the script should run. Default is 1",
        default=1,
        type=int)
    parser.add_argument(
        "-e",
        "--experiments",
        help=
        "Experiment number as string! Use commas like \"2,3,4\" to run multiple experiments. Runs experiment \"1\" by default",
        default="1")
    parser.add_argument("-n","--num_train_steps",
                        help="Number of train steps. Default is 100",
                        default=100,
                        type=int)
    parser.add_argument(
        "--comment",
        help="Optional. Add a Comment to the logfile for internal reference.",
        default="No Comment")
    args = parser.parse_args()
    return args


def main(args):
    args = parse_args(args)

    #Initialise the TPUs if they are used
    if args.use_tpu == 1:
        use_tpu = True
        tpu_address = tpu_init(args.tpu_ip)
        logger.info("Using TPU")
    else:
        use_tpu = False
        tpu_address = None
        logger.info("Using GPU")

    for repeat in range(args.repeats):
        run_experiment(args.experiments, use_tpu, tpu_address, repeat+1, args.num_train_steps,
                       args.username, args.comment)
        logger.info(f'*** Completed repeats {repeat + 1}')


if __name__ == "__main__":
    main(sys.argv[1:])

