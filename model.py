import torch.nn as nn
import torch

class LSTMTextClassifier(nn.Module):
    """
    Parameters
    ----------
    vocab_size          input to Embedding layer, size of vocab
    embedding_dim       
    hidden_layer_sizes  size of hidden states in the LSTM
    num_classes         Number of classes in model out    (2)
    dr                  Dropout rate
    llm                 llm for embedding if one
    """

    def __init__(self, 
                 vocab_size=0, 
                 embedding_dim=0, 
                 hidden_layer_sizes=[100], 
                 num_classes=2, 
                 dr=0.25, 
                 llm=None,):
        super(LSTMTextClassifier, self).__init__()
        self.num_layers = 3
        self.num_classes = num_classes

        self.embedding = nn.Embedding(vocab_size,embedding_dim) 
        self.dropout = nn.Dropout(dr)

        self.LSTM = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_layer_sizes[0],
                            batch_first=True,
                            num_layers=self.num_layers,
                            bidirectional =True,
                            dropout=dr)                                     

        self.pool = nn.AdaptiveMaxPool1d(1)               

        self.fc = nn.Linear(hidden_layer_sizes[0]*2, self.num_classes)         

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight.data)
            module.bias.data.zero_()
        elif isinstance(module, nn.LSTM):
            for i in range(self.num_layers):
                torch.nn.init.xavier_uniform_(module.all_weights[i][0])
                torch.nn.init.xavier_uniform_(module.all_weights[i][1])
        elif isinstance(module, nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight.data)

    def from_embedding(self, embedded):
        # forward pass (dropout, lstm, pool, fc)
        dropouted = self.dropout(embedded)
        output, (hn, cn) = self.LSTM(dropouted)

        # pool lstm outputs
        pooled = self.pool(output.permute(0,2,1)).squeeze()
        if pooled.dim() != 2:
            pooled = pooled.unsqueeze(0)

        logits = self.fc(pooled)
        return logits

    def forward(self, data, mask=None):
        return self.from_embedding(data.unsqueeze(1))


class LLMLSTMTextClassifier(LSTMTextClassifier):
    ''' LLM implementation of LSTM, utilizing LLM embedding layer'''
    def __init__(self, llm, *args, **kwargs,):
        super().__init__(*args, **kwargs)
        self.llm = llm

    def forward(self, data, mask= None):
        # embed with LLM, then pass above
        data = self.llm((data * mask) if mask else data).last_hidden_state
        return self.from_embedding(data)

#-----------------------------------------------------------------------------
class CNNTextClassifier(nn.Module):
    """
    vocab_size          input to Embedder
    embedding_dim 
    num_classes         Number of classes in model out    (2)
    dr                  Dropout rate
    hidden_layer_sizes  The widths for each set of filters
    num_filters         Number of filters for each width
    num_conv_layers     Number of convolutional layers (conv + pool)
    intermediate_pool_size -- 3
    llm for embedding if one
    """
    def __init__(self, vocab_size=0, 
                       embedding_dim=0,
                       num_classes=2,
                       dr=0.2,
                       hidden_layer_sizes=[4,3,2], #filter_widths 
                       num_filters=100, 
                       num_conv_layers=2,
                       intermediate_pool_size=3,
                       llm=None,
                       **kwargs):
        super(CNNTextClassifier, self).__init__(**kwargs)

        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.filter_modules = nn.ModuleList()

        for w in hidden_layer_sizes:
            # Each filter width requires a sequential chain of CNN_BLOCKS
            cnn_blocks = nn.Sequential()

            for n in range(num_conv_layers):
                # A CNN_BLOCK IN EACH LAYER
                cnn_blocks.append(nn.Dropout(dr))

                #emb_dim first pass, num_filters subsequent
                in_dim = embedding_dim if n == 0 else num_filters

                # (embed_dim, num_filters) -> (num_filters, num_filters)
                cnn_blocks.append(nn.Conv1d(in_channels= in_dim, 
                                           out_channels=num_filters,
                                           kernel_size=w, ))            
                cnn_blocks.append(nn.ReLU())

                if n == num_conv_layers -1:
                    # final layer output to 1
                    cnn_blocks.append(nn.AdaptiveMaxPool1d(output_size=1))                  
                else:
                    # intermediate layers
                    cnn_blocks.append(nn.MaxPool1d(kernel_size=(intermediate_pool_size,)))  
                
            self.filter_modules.append(cnn_blocks)
        
        self.fc = nn.Linear(num_filters * len(hidden_layer_sizes) , out_features=num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            torch.nn.init.xavier_uniform_(module.weight.data)

    def from_embedding(self, embedded):   
        # forward pass (from the outputs of the embedding)
        chain_in = embedded.permute(0,2,1)

        #Chain through CNN blocks
        all_filters = []
        for i,chain in enumerate(self.filter_modules):
            chain_out = chain(chain_in)
            all_filters.append(chain_out.squeeze())
        
        all_filters = torch.cat(all_filters,dim=-1)
        # pool outputs
        if all_filters.dim() != 2:
            all_filters = all_filters.unsqueeze(0)

        logits = self.fc(all_filters)
        logits = torch.sigmoid(logits)
        return logits
    
    def forward(self, data, mask=None):
        return self.from_embedding(data.unsqueeze(1))



class LLMCNNTextClassifier(CNNTextClassifier):
    ''' implementation of the CNN with llm embeddings layer '''
    def __init__(self, llm, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = llm

    def forward(self, data, mask= None):
        # embed with llm, then pass above
        data = self.llm((data * mask) if mask else data).last_hidden_state
        return self.from_embedding(data)
