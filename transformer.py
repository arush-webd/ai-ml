import torch
import torch.nn as nn

# Simple Transformer model
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, hidden_dim, num_layers):
        super().__init__()

        # Convert words -> vectors
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=hidden_dim
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Final output layer
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        # x shape: (sequence_length, batch_size)

        embedded = self.embedding(x)

        # Pass through transformer
        output = self.transformer(embedded)

        # Predict next token
        output = self.fc(output)

        return output


# Example usage
vocab_size = 1000
embed_size = 64
num_heads = 4
hidden_dim = 128
num_layers = 2

model = SimpleTransformer(
    vocab_size,
    embed_size,
    num_heads,
    hidden_dim,
    num_layers
)

# Dummy input: (sequence_length=10, batch_size=2)
x = torch.randint(0, vocab_size, (10, 2))

# Forward pass
y = model(x)

print(y.shape)  # (10, 2, 1000)trans
