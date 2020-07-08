import torch

class CrossBatchMemory(torch.nn.Module):
    def __init__(self, loss, embedding_size, memory_size=1024, miner=None):
        super().__init__()
        self.loss = loss
        self.miner = miner
        self.memory_size = memory_size
        self.embedding_memory = torch.zeros(self.memory_size, embedding_size)
        self.label_memory = torch.zeros(self.memory_size).long()
        self.has_been_filled = False
        self.queue_idx = 0

    def forward(self, embeddings, labels, input_indices_tuple=None):
        assert embeddings.size(0) <= self.embedding_memory.size(0)
        batch_size = embeddings.size(0)
        labels = labels.to(embeddings.device)
        self.embedding_memory = self.embedding_memory.to(embeddings.device)
        self.label_memory = self.label_memory.to(labels.device)
        self.add_to_memory(embeddings, labels, batch_size)
        
        if not self.has_been_filled:
            E_mem = self.embedding_memory[:self.queue_idx]
            L_mem = self.label_memory[:self.queue_idx] 
        else:
            E_mem = self.embedding_memory
            L_mem = self.label_memory

        combined_embeddings = torch.cat([embeddings, E_mem], dim=0)
        combined_labels = torch.cat([labels, L_mem], dim=0)
        loss = self.loss(combined_embeddings, combined_labels)
        return loss, combined_labels

    def add_to_memory(self, embeddings, labels, batch_size):
        end_idx = ((self.queue_idx + batch_size - 1) % (self.memory_size)) + 1

        if end_idx > self.queue_idx:
            self.embedding_memory[self.queue_idx:end_idx] = embeddings.detach()
            self.label_memory[self.queue_idx:end_idx] = labels.detach()            
        else:
            se = self.memory_size-self.queue_idx
            self.embedding_memory[self.queue_idx:] = embeddings[:se].detach()
            self.embedding_memory[:end_idx] = embeddings[se:].detach()
            self.label_memory[self.queue_idx:] = labels[:se].detach()
            self.label_memory[:end_idx] = labels[se:].detach()
            

        prev_queue_idx = self.queue_idx
        self.queue_idx = (self.queue_idx + batch_size) % self.memory_size

        if (not self.has_been_filled) and (self.queue_idx <= prev_queue_idx):
            self.has_been_filled = True