import torch

class SGNS(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size

        self.hidden_emb = torch.nn.Embedding(self.vocab_size, self.embedding_size)
        self.target_emb = torch.nn.Embedding(self.vocab_size, self.embedding_size)

        default_w = 0.5 / self.embedding_size
        self.hidden_emb.weight.data.uniform_(-default_w, default_w)
        self.target_emb.weight.data.uniform_(-default_w, default_w)
        
    def forward(self, x, t):
        center_embedding = self.hidden_emb(x)      
        target_embedding = self.target_emb(t)
        
        logit = torch.bmm(target_embedding.unsqueeze(1), center_embedding.unsqueeze(2))
        logit = logit.squeeze().reshape([-1])
        
        return logit