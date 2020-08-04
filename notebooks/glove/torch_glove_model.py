import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class GloveModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, vocab):
        super(GloveModel, self).__init__()

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )
        self.context_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )
        self.token_bias = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=1
        )
        self.context_bias = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=1
        )

        self.vocab = vocab


    def forward(self, token, context_token):
        w_i, b_i = self.token_embedding(token), self.token_bias(token)
        w_j, b_j = self.context_embedding(context_token), self.context_bias(context_token)
        out = (w_i*w_j).sum(dim=1, keepdim=True) + b_i + b_j

        return out.view(-1)

    def loss(self, output, target, x_max=100, alpha=0.75):
        J = torch.sum(self.weight_function(target, alpha, x_max)*(output - torch.log(target))**2)
        return J


    def weight_function(self, target, alpha, x_max):

        weight = (target/x_max)**alpha * (target < x_max) + (target >= x_max)
        return weight

    def __combine_embeddings__(self):
        self.eval()
        return self.token_embedding.weight.data + self.context_embedding.weight.data

    def __embeddings__(self):
        return WordVector(self.__combine_embeddings__().cpu().numpy(), self.vocab)

class WordVector(object):

    def __init__(self, embeddings, vocab):
        self.embeddings = embeddings
        self.vocab = vocab

    def __getitem__(self, word_str_or_id):
        if isinstance(word_str_or_id, str):
            return self.embeddings[self.vocab[word_str_or_id]]
        elif isinstance(word_str_or_id, int):
            return self.embeddings[word_str_or_id]

    def __len__(self):
        return self.embeddings.shape


def train(data_loader, model, optimizer, epoch, x_max=100, alpha=0.75, 
            device='cpu', progress_bar=None):
    """Train Model and return the loss

    Args:
        model (nn.Module): Pytorch Model instance
        batch_generator (generator): Generator of training batches
        optimizer (torch.optim.Optimizer): Pytorch optimizer instance
        epoch (int): actual epoch
        device (str, optional): which device. Defaults to 'cpu'.
        progress_bar (tqdm.tqdm): progress bar for training progress
    """
    model.train()
    running_loss = 0.0

    for batch_index, batch_dict in enumerate(data_loader):
        optimizer.zero_grad()
        # Get Training data
        token, context, count = batch_dict.values()
        # Compute Output
        ouput = model(token, context)
        # Calculate loss
        loss = model.loss(ouput, count, x_max=x_max, alpha=alpha)
        loss_t = loss.item()
        running_loss += (loss_t - running_loss) / (batch_index + 1)
        # Backward
        loss.backward()
        # Take gradient step
        optimizer.step()

        progress_bar.set_postfix(loss=running_loss, epoch=epoch + 1)
        progress_bar.update()
    
    return running_loss