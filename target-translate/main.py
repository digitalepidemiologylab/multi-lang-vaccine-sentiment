from base_model import BaseModel
import csv
import logging
import os
import random
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from transformers import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import time
import argparse
import uuid
import json


logger = logging.getLogger(__name__)

class BERTModel(BaseModel):
    def __init__(self, args):
        super().__init__()
        self.estimator = None
        self.label_mapping = None
        self.train_examples = None
        self.num_train_optimization_steps = None
        # Hyperparams
        self.max_seq_length = args.max_seq_length
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size
        # Initial learning rate for Adam optimizer
        self.learning_rate = args.learning_rate
        self.num_epochs = args.epochs
        # Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.
        self.warmup_steps = args.warmup_steps
        self.no_cuda = args.no_cuda
        # Number of updates steps to accumulate before performing a backward/update pass.
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.seed = args.seed
        # Use 16 bit float precision (instead of 32bit)
        self.fp16 = args.fp16
        # Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.
        # 0 (default value): dynamic loss scaling. Positive power of 2: static loss scaling value.
        self.loss_scale = args.loss_scale
        # Meta params
        self.write_test_output = args.write_test_output
        self.output_attentions = args.output_attentions
        self.eval_after_epoch = args.eval_after_epoch
        self.username = args.username
        # model
        self.model_type = args.model_type
        # paths
        self.train_data_path = os.path.join(args.data_path, args.train_data, 'train.tsv')
        self.dev_data_path = os.path.join(args.data_path, args.dev_data, 'dev.tsv')
        self.test_data_path = os.path.join(args.data_path, args.test_data, 'test.tsv')
        self.other_path = args.other_path
        self.default_output_folder = 'output'
        self.output_path = self.generate_output_path(args.output_path)
        self.model_path = os.path.join(self.other_path, 'bert')
        self.all_args = vars(args)

    def generate_output_path(self, output_path):
        if output_path is None:
            output_path = os.path.join(self.default_output_folder, f"{time.strftime('%Y_%m_%d-%-H_%M_%S')}-{str(uuid.uuid4())[:4]}-{self.username}")
        return output_path

    def create_dirs(self):
        for _dir in [self.output_path]:
            logger.info(f'Creating directory {_dir}')
            os.makedirs(_dir)

    def train(self):
        # Setup
        self._setup_bert()

        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        if self.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
            self.optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=self.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if self.loss_scale == 0:
                self.optimizer = FP16_Optimizer(self.optimizer, dynamic_loss_scale=True)
            else:
                self.optimizer = FP16_Optimizer(self.optimizer, static_loss_scale=self.loss_scale)
        else:
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.num_train_optimization_steps)

        # Run training
        global_step = 0
        tr_loss = 0
        train_features = self.convert_examples_to_features(self.train_examples)
        logger.debug("***** Running training *****")
        logger.debug("  Num examples = %d", len(self.train_examples))
        logger.debug("  Batch size = %d", self.train_batch_size)
        logger.debug("  Num steps = %d", self.num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.train_batch_size)
        loss_vs_time = []
        for epoch in range(int(self.num_epochs)):
            self.model.train()
            nb_tr_examples, nb_tr_steps = 0, 0
            epoch_loss = 0
            pbar = tqdm(train_dataloader)
            for step, batch in enumerate(pbar):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss, logits = self.model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids, labels=label_ids)
                if self.n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps
                if self.fp16:
                    self.optimizer.backward(loss)
                else:
                    loss.backward()
                loss = loss.item()
                tr_loss += loss
                epoch_loss += loss
                if step > 0:
                    pbar.set_description("Loss: {:8.4f} | Average loss/it: {:8.4f}".format(loss, epoch_loss/step))
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
            # evaluate model
            if self.eval_after_epoch:
                self.model.eval()
                nb_train_steps, nb_train_examples = 0, 0
                train_accuracy, train_loss = 0, 0
                for input_ids, input_mask, segment_ids, label_ids in tqdm(train_dataloader, desc="Evaluating"):
                    input_ids = input_ids.to(self.device)
                    input_mask = input_mask.to(self.device)
                    segment_ids = segment_ids.to(self.device)
                    label_ids = label_ids.to(self.device)
                    with torch.no_grad():
                        loss, logits = self.model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids, labels=label_ids)
                    train_accuracy += self.accuracy(logits.to('cpu').numpy(), label_ids.to('cpu').numpy())
                    train_loss += loss.mean().item()
                    nb_train_examples += input_ids.size(0)
                    nb_train_steps += 1
                train_loss = train_loss / nb_train_steps
                train_accuracy = 100 * train_accuracy / nb_train_examples
                print("{bar}\nEpoch {}:\nTraining loss: {:8.4f} | Training accuracy: {:.2f}%\n{bar}".format(epoch+1, train_loss, train_accuracy, bar=80*'='))

        # Save model
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-self
        output_model_file = os.path.join(self.output_path, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(self.output_path, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())
        args_output_file = os.path.join(self.output_path, 'args.json')
        with open(args_output_file, 'w') as f:
            json.dump(self.all_args, f)


    def test(self):
        # Setup
        self._setup_bert(setup_mode='test')
        # Run test
        eval_examples = self.processor.get_dev_examples(self.dev_data_path)
        eval_features = self.convert_examples_to_features(eval_examples)
        logger.debug("***** Running evaluation *****")
        logger.debug("  Num examples = %d", len(eval_examples))
        logger.debug("  Batch size = %d", self.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.eval_batch_size)
        self.model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        result = {'prediction': [], 'label': [], 'text': []}
        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            label_ids = label_ids.to(self.device)
            tmp_eval_loss, logits = self.model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids, labels=label_ids)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            result['prediction'].extend(np.argmax(logits, axis=1).tolist())
            result['label'].extend(label_ids.tolist())
            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
        eval_loss = eval_loss / nb_eval_steps
        label_mapping = self.get_label_mapping()
        result_out = self.performance_metrics(result['label'], result['prediction'], label_mapping=label_mapping)
        if self.write_test_output:
            test_output = self.get_full_test_output(result['prediction'], result['label'], label_mapping=label_mapping,
                    test_data_path=self.dev_data_path)
            result_out = {**result_out, **test_output}
        return result_out

    def save_results(self, results):
        result_path = os.path.join(self.output_path, 'results.json')
        logger.info(f'Writing output results to {result_path}...')
        with open(result_path, 'w') as f:
            json.dump(results, f)

    def predict(self, data):
        """Predict data (list of strings)"""
        # Setup
        self._setup_bert(setup_mode='predict', data=data)
        # Run predict
        predict_examples = self.processor.get_test_examples(data)
        predict_features = self.convert_examples_to_features(predict_examples)
        all_input_ids = torch.tensor([f.input_ids for f in predict_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in predict_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in predict_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in predict_features], dtype=torch.long)
        predict_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        predict_sampler = SequentialSampler(predict_data)
        predict_dataloader = DataLoader(predict_data, sampler=predict_sampler, batch_size=self.eval_batch_size)
        self.model.eval()
        result = []
        for input_ids, input_mask, segment_ids, label_ids in predict_dataloader:
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            output = self.model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
            logits = output[0]
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            probabilities = probabilities.detach().cpu().numpy()
            res = self.format_predictions(probabilities, label_mapping=self.label_mapping)
            result.extend(res)
        return result

    def fine_tune(self):
        raise NotImplementedError

    def _setup_bert(self, setup_mode='train', data=None):
        # Create necessary dirctory structure
        if setup_mode == 'train':
            self.create_dirs()

        # GPU config
        self.device = torch.device("cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
        self.n_gpu = torch.cuda.device_count()
        if self.no_cuda:
            self.n_gpu = 0
        if setup_mode == 'train':
            logger.info("Initialize BERT: device: {}, n_gpu: {}, distributed training: {}, 16-bits training: {}".format(self.device, self.n_gpu, False, self.fp16))
        if self.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(self.gradient_accumulation_steps))
        self.train_batch_size = self.train_batch_size // self.gradient_accumulation_steps

        # seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(self.seed)

        # label mapping
        if setup_mode == 'train':
            self.label_mapping = self.set_label_mapping()
        elif setup_mode in ['test', 'predict']:
            self.label_mapping = self.get_label_mapping()

        # Build model
        self.processor = SentimentClassificationProcessor(self.train_data_path, self.label_mapping)
        num_labels = len(self.label_mapping)
        self.do_lower_case = 'uncased' in self.model_type
        self.tokenizer = BertTokenizer.from_pretrained(self.model_type, do_lower_case=self.do_lower_case)
        if setup_mode == 'train':
            self.train_examples = self.processor.get_train_examples(self.train_data_path)
            self.num_train_optimization_steps = int(len(self.train_examples) / self.train_batch_size / self.gradient_accumulation_steps) * self.num_epochs

        # Prepare model
        if setup_mode == 'train':
            # if self.fine_tune_path:
            #     logger.info('Loading fine-tuned model {} of type {}...'.format(self.fine_tune_path, self.model_type))
            #     config = BertConfig(os.path.join(self.fine_tune_path, CONFIG_NAME))
            #     weights = torch.load(os.path.join(self.fine_tune_path, WEIGHTS_NAME))
            #     self.model = BertForSequenceClassification.from_pretrained(self.model_type, cache_dir=self.model_path, num_labels=num_labels, state_dict=weights)
            # else:
            #     logger.info('Loading pretrained model {}...'.format(self.model_type))
            #     self.model = BertForSequenceClassification.from_pretrained(self.model_type, cache_dir=self.model_path, num_labels = num_labels)
            self.model = BertForSequenceClassification.from_pretrained(self.model_type, cache_dir=self.model_path, num_labels = num_labels)
            if self.fp16:
                self.model.half()
        else:
            # Load a trained model and config that you have trained
            self.model = BertForSequenceClassification.from_pretrained(self.output_path)
        self.model.to(self.device)
        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def accuracy(self, out, labels):
        outputs = np.argmax(out, axis=1)
        return np.sum(outputs == labels)

    def convert_examples_to_features(self, examples):
        """Loads a data file into a list of `InputBatch`s."""
        features = []
        for (ex_index, example) in enumerate(examples):
            tokens_a = self.tokenizer.tokenize(str(example.text_a))
            tokens_b = None
            if example.text_b:
                tokens_b = self.tokenizer.tokenize(str(example.text_b))
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                self._truncate_seq_pair(tokens_a, tokens_b, self.max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > self.max_seq_length - 2:
                    tokens_a = tokens_a[:(self.max_seq_length - 2)]
            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0   0   0   0  0     0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambigiously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            if tokens_b:
                tokens += tokens_b + ["[SEP]"]
                segment_ids += [1] * (len(tokens_b) + 1)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)
            # Zero-pad up to the sequence length.
            padding = [0] * (self.max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            assert len(input_ids) == self.max_seq_length
            assert len(input_mask) == self.max_seq_length
            assert len(segment_ids) == self.max_seq_length
            label_id = self.label_mapping[example.label]
            if ex_index < 5:
                logger.debug("*** Example ***")
                logger.debug("guid: %s" % (example.guid))
                logger.debug("tokens: %s" % " ".join(
                        [str(x) for x in tokens]))
                logger.debug("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.debug("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.debug(
                        "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.debug("label: %s (id = %d)" % (example.label, label_id))
            features.append(
                    InputFeatures(input_ids=input_ids,
                                  input_mask=input_mask,
                                  segment_ids=segment_ids,
                                  label_id=label_id))
        return features

class InputExample():
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class SentimentClassificationProcessor():
    """Processor for the sentiment classification data set."""
    def __init__(self, train_path, labels):
        self.labels = labels
        self.train_path = train_path

    def _read_csv(self, input_file):
        return pd.read_csv(input_file, delimiter='\t', header=None)

    def train_validation_split(self, validation_size=0.1):
        with open(self.train_path) as f:
            num_train_examples =  sum(1 for line in f) - 1
            ids = np.arange(num_train_examples)
            np.random.shuffle(ids)
            split_id = int(num_train_examples*validation_size)
            return ids[:split_id], ids[split_id:]

    def get_train_examples(self, data_path):
        """See base class."""
        return self._create_examples(self._read_csv(data_path), "train")

    def get_dev_examples(self, data_path):
        """See base class."""
        return self._create_examples(self._read_csv(data_path), "dev")

    def get_test_examples(self, data_path):
        """See base class."""
        return self._create_examples(self._read_csv(data_path), "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        if isinstance(lines, list):
            lines = pd.DataFrame({'text': lines})
        for (i, line) in lines.iterrows():
            guid = "%{}-{}".format(set_type, i)
            if set_type in ['train', 'dev']:
                text = line[3]
                label = line[1]
            elif set_type == 'test':
                text = ''
                label = list(self.labels)[0]
            else:
                raise Exception(f'Unknown set type {set_type}')
            examples.append(InputExample(guid=guid, text_a=text, text_b=None, label=label))
        return examples

def parse_args(args):
    parser = argparse.ArgumentParser(description='Target translate method')
    parser.add_argument('-u', '--username', dest='username', help='Optional. Username is used in the directory name and in the logfile', default='Anonymous')
    parser.add_argument('--train-data', dest='train_data', help='Data folder with train.tsv file', default='cb-annot-en')
    parser.add_argument('--dev-data', dest='dev_data', help='Data folder with dev.tsv file', default='cb-annot-en')
    parser.add_argument('--test-data', dest='test_data', help='Data folder with test.tsv file', default='cb-annot-en')
    parser.add_argument('--output-path', dest='output_path', help='Path to all output/results', default=None)
    parser.add_argument('--other-path', dest='other_path', help='Path to other resources', default='other')
    parser.add_argument('--data-path', dest='data_path', help='Path to data', default='../data')
    parser.add_argument('--epochs', help='Number of train epochs', default=3, type=int)
    parser.add_argument('--gradient-accumulation-steps', dest='gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--comment', help='Optional. Add a Comment to the logfile for internal reference.', default='No Comment')
    parser.add_argument('--max-seq-length', dest='max_seq_length', default=128, type=int)
    parser.add_argument('--train-batch-size', dest='train_batch_size', default=32, type=int)
    parser.add_argument('--eval-batch-size', dest='eval_batch_size', default=32, type=int)
    parser.add_argument('--lr', dest='learning_rate', default=5e-5, type=float)
    parser.add_argument('--warmup-steps', dest='warmup_steps', default=100, type=int)
    parser.add_argument('--no-cuda', dest='no_cuda', action='store_true', default=False)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--fp16', action='store_true', help='Use 16 bit float precision', default=False)
    parser.add_argument('--loss-scale', dest='loss_scale', type=int, default=0, help='Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.')
    parser.add_argument('--write-test-output', dest='write_test_output', action='store_true', default=False, help='Writes full test output predictions to csv')
    parser.add_argument('--output-attentions', dest='output_attentions', action='store_true', default=False, help='Returns attentions')
    parser.add_argument('--eval-after-epoch', dest='eval_after_epoch', action='store_true', default=False, help='Evaluate after every epoch')
    parser.add_argument('--model-type', dest='model_type', default='bert-base-uncased', help='Model type')
    args = parser.parse_args()
    return args

def main(args):
    # Parse args
    args = parse_args(args)

    # Run experiments
    bert = BERTModel(args)
    bert.train()
    results = bert.test()
    bert.save_results(results)

if __name__ == "__main__":
    main(sys.argv[1:])
