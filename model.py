import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()

        self.hidden_dim = hidden_size
        
        # embedding layer
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)

        # lstm
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers = num_layers, batch_first = True)

        # fc
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        
        embeddings = self.word_embeddings(captions[:,:-1])   # captions[:,:-1] remove the <end> token
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.fc(hiddens)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sampled_ids = []

        for i in range(20):                                    # maximum sampling length
            hiddens, states = self.lstm(inputs, states)        # (batch_size, 1, hidden_size), 
            outputs = self.fc(hiddens.squeeze(1))

            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted.tolist()[0])
            inputs = self.word_embeddings(predicted)
            inputs = inputs.unsqueeze(1)                     

        return sampled_ids