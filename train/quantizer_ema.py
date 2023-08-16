class QuantizerEMA(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(QuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._n_embeddings = n_embeddings

        self._embedding = nn.Embedding(self._n_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        # TODO: not sure if this is the right way to solve DDP issues:
        self._embedding.weight.requires_grad=False
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(n_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(n_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        # TODO: not sure if this is the right way to solve DDP issues:
        self._ema_w.requires_grad = False

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.reshape(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._n_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).reshape(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._n_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            # TODO: not sure if this is the right way to solve DDP issues:
            self._ema_w.requires_grad = False

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
            # TODO: not sure if this is the right way to solve DDP issues:
            self._embedding.weight.requires_grad=False

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        latent_loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, latent_loss, perplexity, encoding_indices
