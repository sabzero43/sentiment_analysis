import random
import time
import datetime
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


class Model:
    """
    Help on  class for pyTorch BERT sentiment classification model
        Args:
            model:      HuggingFace pyTorch BertForSequenceClassification model
                    (https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification)
            optimizer:  optimizer for model
            scheduler:  (`optional`) scheduler for model
    """

    def __init__(self, model, optimizer, scheduler=False):
        """
        Construct a wrapper for pyTorch BERT sentiment classification model
            Args:
                model:      HuggingFace pyTorch BertForSequenceClassification model
                        (https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification)
                optimizer:  optimizer for model
                scheduler:  (`optional`) scheduler for model
        """
        self.training_stats = []
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Will save the best model
        self.best_score = 0
        self.best_epoch = 0
        self.best_params = model.state_dict()

        # Set the seed value all over the place to make this reproducible.
        SEED = 42

        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    def fit(self, train_dataloader, validation_dataloader, epochs=1, early_stopping=False, early_stopping_delay=1):
        """
        Trains and validates the model
            Args:
                train_dataloader:       pyTorch DataLoader with train data for BERT classification model
                validation_dataloader:  pyTorch DataLoader with validation data for BERT classification model
                epochs:                 number of training epochs (default 1)
                early_stopping:         stop the training in case of overfitting (default False)
                early_stopping_delay:   delay of early stopping
        """
        optimizer = self.optimizer
        scheduler = self.scheduler
        model = self.model
        model.cuda()

        # We'll store a number of quantities such as training and validation loss,
        # validation accuracy, and timings.

        # Measure the total training time for the whole run.
        total_t0 = time.time()

        # For each epoch...
        for epoch_i in range(epochs):
            # ========================================
            #               Training
            # ========================================

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Training loop and loss count
            total_train_loss = self._train(model, optimizer, scheduler, train_dataloader, epoch_i + 1)

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)

            # Measure how long this epoch took.
            training_time = self._format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epoch took: {:}".format(training_time))

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")

            t0 = time.time()

            total_eval_accuracy, total_eval_loss = self._validation(model, validation_dataloader)

            # Report the final accuracy for this validation run.
            avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(validation_dataloader)

            # Measure how long the validation run took.
            validation_time = self._format_time(time.time() - t0)

            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            # Record all statistics from this epoch.
            self.training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accuracy': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )

            # Save model if the best
            if (1 / avg_val_loss) > self.best_score:
                self.best_params = model.state_dict()
                self.best_score = 1 / avg_val_loss
                self.best_epoch = epoch_i

            # Early stop in case of overfitting
            if early_stopping and epoch_i - self.best_epoch > early_stopping_delay:
                print("Model overfited!")
                print("Early stop!")
                break

        print("")
        print("Training complete!")

        print("Total training took {:} (h:mm:ss)".format(self._format_time(time.time() - total_t0)))

        model.load_state_dict(self.best_params)

    def predict(self, prediction_dataloader):
        """
        Predict the class labels based on prediction_dataloader
            Args:
                prediction_dataloader: pyTorch DataLoader with data for prediction

            Returns:
                Logits and class labels for each sentences from prediction_dataloader
        """
        model = self.model
        device = self.device

        model.eval()
        test_preds, test_labels = [], []

        for batch in log_progress(prediction_dataloader, every=50, name='Prediction in process...'):
            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)

            # When using .no_grad (), the model will not count and store gradients.
            # This will speed up the process of predicting labels for the test data.
            with torch.no_grad():
                logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            # Moving logits and class labels to the CPU for further work
            logits = logits[0].detach().cpu().numpy()

            # Save predicted classes and ground truth
            batch_preds = np.argmax(logits, axis=1)
            test_preds.extend(batch_preds)

        return test_preds

    def show_training_stats(self):
        """
        Print training results:
            Table with training and validation metrics for each epoch
            Plot with training and validation loss
        """
        # Create a DataFrame from our training statistics.
        df_stats = pd.DataFrame(data=self.training_stats)

        # Use plot styling from seaborn.
        sns.set(style='darkgrid')

        # Increase the plot size and font size.
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (12, 6)

        # Plot the learning curve.
        plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
        plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

        # Label the plot.
        plt.title("Training & Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.xticks([1, 2, 3, 4])

        plt.show()
        print('\n')

        # Display floats with two decimal places.
        pd.set_option('precision', 2)

        # Label best model
        df_stats['Best Model'] = False
        df_stats.loc[self.best_epoch, 'Best Model'] = True

        # Use the 'epoch' as the row index.
        df_stats = df_stats.set_index('epoch')

        # Display the table.
        return df_stats

    @staticmethod
    def _format_time(elapsed):
        """
        Takes a time in seconds and returns a string hh:mm:ss
        """
        # Round to the nearest second.
        elapsed_rounded = int(round(elapsed))
        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    @staticmethod
    def _flat_accuracy(preds, labels):
        """
        Helper function to calculate the accuracy of our predictions vs labels
        Args:
            preds:  model prediction
            labels: actual classes

        Returns:
            accuracy of predictions vs labels
        """
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def _train(self, model, optimizer, scheduler, train_dataloader, epoch_i):
        """
        Train loop
        Args:
            model:              HuggingFace pyTorch BertForSequenceClassification model
                        (https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification)
            optimizer:          optimizer for model
            scheduler:          scheduler for model (even if scheduler == False)
            train_dataloader:   pyTorch DataLoader with train data
            epoch_i:            epoch number

        Return:
            Total train loss

        """
        # Reset the total loss for this epoch.
        total_train_loss = 0

        model.train()

        # For each batch of training data...
        for batch in log_progress(train_dataloader, every=50, name='Epoch {} in progress...'.format(epoch_i)):
            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # It returns different numbers of parameters depending on what arguments
            # given and what flags are set. For our usage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            loss, logits = model(b_input_ids,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask,
                                 labels=b_labels)

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            if scheduler:
                scheduler.step()

        return total_train_loss

    def _validation(self, model, validation_dataloader):
        """
        Validation after training epoch
        Args:
            model:                  HuggingFace pyTorch BertForSequenceClassification model
                            (https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification)
            validation_dataloader:  pyTorch dataloader with validation data
         
        """
        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0

        # Evaluate data for one epoch
        for batch in log_progress(validation_dataloader, every=50, name='Validation in progress...'):
            # Unpack this training batch from our dataloader.
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backpropagation (training).
            with torch.no_grad():
                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                loss, logits = model(b_input_ids,
                                     token_type_ids=None,
                                     attention_mask=b_input_mask,
                                     labels=b_labels)

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += self._flat_accuracy(logits, label_ids)

        return total_eval_accuracy, total_eval_loss


def log_progress(sequence, every=None, size=None, name='Items'):
    """
    https://github.com/kuk/log-progress
    """
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)  # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except BaseException:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )


def train_validation_split(tokenizer, train_text, train_labels):
    """
    Tokenize and split data on train and validation samples
    Args:
        tokenizer:      BERT tokenizer
        train_text:     train sentences to split on
        train_labels:   train labels
    Returns:
        train_inputs, validation_inputs, train_labels, validation_labels, train_masks, validation_masks
    """
    SEED = 42

    input_ids, attention_masks = tokenize(train_text, tokenizer)
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(
        input_ids, train_labels,
        random_state=SEED,
        test_size=0.2,
    )

    train_masks, validation_masks, _, _ = train_test_split(
        attention_masks,
        input_ids,
        random_state=SEED,
        test_size=0.2,
    )

    return train_inputs, validation_inputs, train_labels, validation_labels, train_masks, validation_masks


def tokenize(sentences, tokenizer):
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

    max_len = pd.Series(tokenized_texts).map(lambda x: len(x)).quantile(1)
    max_len = int(max_len)

    # Convert tokens to numbers and add padding
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(
        input_ids,          # sentence tokens
        maxlen=max_len,     # maximum number of tokens
        dtype="long",
        truncating="post",  # cut after
        padding="post"      # add padding after
    )
    attention_masks = [[float(i > 0) for i in seq] for seq in input_ids]

    return input_ids, attention_masks


def batch_data_loader(inputs, masks, labels=False, batch_size=25):
    inputs_tensor = torch.LongTensor(inputs)
    masks_tensor = torch.LongTensor(masks)

    if labels:
        labels_tensor = torch.LongTensor(labels)
        data = TensorDataset(inputs_tensor, masks_tensor, labels_tensor)
        dataloader = DataLoader(
            data,
            shuffle=True,
            batch_size=batch_size
        )
    else:
        data = TensorDataset(inputs_tensor, masks_tensor)
        dataloader = DataLoader(
            data,
            shuffle=False,
            batch_size=batch_size
        )

    return dataloader


def prediction_data_processing(sentences, tokenizer, batch_size=64):
    """
    Prepare data for inference
    Args:
        sentences:  text data
        tokenizer:  BERT tokenizer
        batch_size: size of batches
    Returns:
        Dataloader to be fed into the model
    """
    input_ids, attention_masks = tokenize(sentences, tokenizer)

    prediction_dataloader = batch_data_loader(input_ids, attention_masks, batch_size=batch_size)

    return prediction_dataloader


