import torch
import numpy as np
from nltk.tokenize import sent_tokenize
import os
import nltk
from tqdm import trange
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
from seqeval.metrics import accuracy_score, f1_score

nltk.download('punkt')
import torch.nn as nn
import matplotlib.pyplot as plt

class NER_BioBERT(object):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tag2idx = {'O': 0, 'ID': 1, 'PHI': 2, 'NAME': 3, 'CONTACT': 4, 'DATE': 5, 'AGE': 6, 'PROFESSION': 7, 'LOCATION': 8, 'PAD': 9}
    tag_values = ["O", "ID", "PHI", "NAME", "CONTACT", "DATE", "AGE", "PROFESSION", "LOCATION", "PAD"]

    tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1', do_lower_case=False)
    model = AutoModelForTokenClassification.from_pretrained('dmis-lab/biobert-v1.1', num_labels=len(tag2idx))

    MAX_LEN = 75
    bs = 4

    def __init__(self):
        model_path = "Models/NER_BioBERT.pt"
        if os.path.exists(model_path):
            print("Loading BioBERT model from", model_path)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print("Using pre-trained BioBERT model")
        self.model.to(self.device)

    def perform_NER(self, text):
        self.model.eval()
        list_of_sents = sent_tokenize(text)
        list_of_tuples_by_sent = []

        for sent in list_of_sents:
            tokenized_sentence = self.tokenizer.encode(sent, truncation=True, padding="max_length", max_length=self.MAX_LEN)
            input_ids = torch.tensor([tokenized_sentence]).to(self.device)

            with torch.no_grad():
                output = self.model(input_ids)
            label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)

            tokens = self.tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
            new_tokens, new_labels = [], []
            for token, label_idx in zip(tokens, label_indices[0]):
                if token.startswith("##"):
                    new_tokens[-1] = new_tokens[-1] + token[2:]
                else:
                    new_labels.append(self.tag_values[label_idx])
                    new_tokens.append(token)
            list_of_tuples = [(token, label) for token, label in zip(new_tokens, new_labels)]
            list_of_tuples_by_sent.append(list_of_tuples)

        return list_of_tuples_by_sent

    def tokenize_and_preserve_labels(self, sentence, text_labels):
        tokenized_sentence = []
        labels = []

        for word, label in zip(sentence, text_labels):
            tokenized_word = self.tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)

            tokenized_sentence.extend(tokenized_word)
            labels.extend([label] * n_subwords)

        return tokenized_sentence, labels

    def transform_sequences(self, tokens_labels):
        tokenized_sentences = []
        labels = []
        for sentence in tokens_labels:
            text_labels = [word_label[1] for word_label in sentence]
            sentence_to_feed = [word_label[0] for word_label in sentence]
            a, b = self.tokenize_and_preserve_labels(sentence_to_feed, text_labels)
            tokenized_sentences.append(a)
            labels.append(b)

        input_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_sentences],
                                  maxlen=self.MAX_LEN, dtype="long", value=0.0,
                                  truncating="post", padding="post")

        tags = pad_sequences([[self.tag2idx.get(l) for l in lab] for lab in labels],
                             maxlen=self.MAX_LEN, value=self.tag2idx["PAD"], padding="post",
                             dtype="long", truncating="post")

        return input_ids, tags

    def learn(self, X_train, Y_train, epochs=1):
        tr_masks = [[float(i != 0.0) for i in ii] for ii in X_train]

        tr_inputs = torch.tensor(X_train).type(torch.long)
        tr_tags = torch.tensor(Y_train).type(torch.long)
        tr_masks = torch.tensor(tr_masks).type(torch.long)

        train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.bs)

        FULL_FINETUNING = True
        if FULL_FINETUNING:
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
            ]
        else:
            param_optimizer = list(self.model.classifier.named_parameters())
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

        optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5, eps=1e-8, no_deprecation_warning=True)
        max_grad_norm = 1.0
        total_steps = len(train_dataloader) * epochs

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        loss_values = []

        for _ in trange(epochs, desc="Epoch"):
            self.model.train()
            total_loss = 0

            for step, batch in enumerate(train_dataloader):
                batch = tuple(b.to(self.device) for b in batch)
                b_input_ids, b_input_mask, b_labels = batch
                self.model.zero_grad()
                try:
                    outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                except RuntimeError as e:
                    print(f"CUDA error encountered: {e}")
                    break  # Break out of the training loop on CUDA error
                
                loss = outputs[0]
                loss.backward()
                total_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_loss / len(train_dataloader)
            loss_values.append(avg_train_loss)
            print("Average train loss: {}".format(avg_train_loss))

        plt.figure()
        plt.plot(loss_values, 'b-o', label="training loss")
        plt.title("Learning curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def evaluate(self, X_test, Y_test):
        val_masks = [[float(i != 0.0) for i in ii] for ii in X_test]
        val_inputs = torch.tensor(X_test).type(torch.long)
        val_tags = torch.tensor(Y_test).type(torch.long)
        val_masks = torch.tensor(val_masks).type(torch.long)

        valid_data = TensorDataset(val_inputs, val_masks, val_tags)
        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=self.bs)

        self.model.eval()
        eval_loss, eval_accuracy = 0, 0
        predictions, true_labels = [], []
        for batch in valid_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            eval_loss += outputs[0].mean().item()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.extend(label_ids)

        eval_loss = eval_loss / len(valid_dataloader)
        print("Validation loss: {}".format(eval_loss))
        pred_tags = [self.tag_values[p_i] for p, l in zip(predictions, true_labels) for p_i, l_i in zip(p, l) if self.tag_values[l_i] != "PAD"]
        valid_tags = [self.tag_values[l_i] for l in true_labels for l_i in l if self.tag_values[l_i] != "PAD"]
        print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
        print("Validation F1-Score: {}".format(f1_score(valid_tags, pred_tags, average='weighted')))
        labels = ["ID", "PHI", "NAME", "CONTACT", "DATE", "AGE", "PROFESSION", "LOCATION"]
        print(classification_report(valid_tags, pred_tags, digits=4, labels=labels))

    def save(self, model_path):
        torch.save(self.model.state_dict(), "Models/" + model_path + ".pt")
        print("Saved model to disk")
