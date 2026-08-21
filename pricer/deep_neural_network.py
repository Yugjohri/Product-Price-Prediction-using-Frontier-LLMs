import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.feature_extraction.text import HashingVectorizer


class SparseBatchDataset(Dataset):
    """Keeps the hashed features sparse, densifying only one batch at a time.

    The feature matrix is 5,000-wide and mostly zeros. Densifying it up front
    costs ~8 GB on the full dataset, which does not fit alongside everything
    else; densifying per batch costs ~1 MB. The tensors the model sees are
    identical either way.
    """

    def __init__(self, X_sparse, y):
        self.X = X_sparse.tocsr()
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idxs):
        # Paired with BatchSampler + batch_size=None, idxs is a list of indices.
        rows = self.X[idxs].toarray()
        return torch.from_numpy(rows).float(), self.y[idxs]


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, dropout_prob):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual  # Skip connection
        return self.relu(out)


class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_size, num_layers=10, hidden_size=4096, dropout_prob=0.2):
        super(DeepNeuralNetwork, self).__init__()

        # First layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )

        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for i in range(num_layers - 2):
            self.residual_blocks.append(ResidualBlock(hidden_size, dropout_prob))

        # Output layer
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.input_layer(x)

        for block in self.residual_blocks:
            x = block(x)

        return self.output_layer(x)


class DeepNeuralNetworkRunner:
    def __init__(self, train, val):
        self.train_data = train
        self.val_data = val
        self.vectorizer = None
        self.model = None
        self.device = None
        self.loss_function = None
        self.optimizer = None
        self.scheduler = None
        self.train_dataset = None
        self.train_loader = None
        self.y_mean = None
        self.y_std = None

        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    def setup(self):
        self.vectorizer = HashingVectorizer(n_features=5000, stop_words="english", binary=True)

        train_documents = [item.summary for item in self.train_data]
        self.X_train_sparse = self.vectorizer.fit_transform(train_documents)
        y_train_np = np.array([float(item.price) for item in self.train_data])
        self.y_train = torch.FloatTensor(y_train_np).unsqueeze(1)

        val_documents = [item.summary for item in self.val_data]
        X_val_np = self.vectorizer.transform(val_documents)
        self.X_val = torch.FloatTensor(X_val_np.toarray())
        y_val_np = np.array([float(item.price) for item in self.val_data])
        self.y_val = torch.FloatTensor(y_val_np).unsqueeze(1)

        y_train_log = torch.log(self.y_train + 1)
        y_val_log = torch.log(self.y_val + 1)
        self.y_mean = y_train_log.mean()
        self.y_std = y_train_log.std()
        self.y_train_norm = (y_train_log - self.y_mean) / self.y_std
        self.y_val_norm = (y_val_log - self.y_mean) / self.y_std

        self.model = DeepNeuralNetwork(self.X_train_sparse.shape[1])
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Deep Neural Network created with {total_params:,} parameters")

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"Using {self.device}")

        self.model.to(self.device)
        self.loss_function = nn.L1Loss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10, eta_min=0)

        self.train_dataset = SparseBatchDataset(self.X_train_sparse, self.y_train_norm)
        # batch_size=None because the BatchSampler already hands over index
        # lists; the dataset densifies each batch itself.
        self.train_loader = DataLoader(
            self.train_dataset,
            sampler=BatchSampler(RandomSampler(self.train_dataset), batch_size=64, drop_last=False),
            batch_size=None,
        )

    def train(self, epochs=5):
        for epoch in range(1, epochs + 1):
            self.model.train()
            train_losses = []

            for batch_X, batch_y in tqdm(self.train_loader):
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.loss_function(outputs, batch_y)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                train_losses.append(loss.item())

            # Validation, chunked: a single forward pass over a large validation
            # set allocates activations for every row at once and exhausts GPU
            # memory. Chunking changes nothing about the result.
            self.model.eval()
            with torch.no_grad():
                chunks = []
                for start in range(0, self.X_val.shape[0], 512):
                    batch = self.X_val[start : start + 512].to(self.device)
                    chunks.append(self.model(batch))
                val_outputs = torch.cat(chunks)

                val_loss = self.loss_function(val_outputs, self.y_val_norm.to(self.device))

                # Convert back to original scale for meaningful metrics
                val_outputs_orig = torch.exp(val_outputs * self.y_std + self.y_mean) - 1
                mae = torch.abs(val_outputs_orig - self.y_val.to(self.device)).mean()

            avg_train_loss = np.mean(train_losses)
            print(f"Epoch [{epoch}/{epochs}]")
            print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss.item():.4f}")
            print(f"Val mean absolute error: ${mae.item():.2f}")
            print(f"Learning rate: {self.scheduler.get_last_lr()[0]:.6f}")

            self.scheduler.step()

    def save(self, path):
        # y_mean/y_std are part of the model, not of the run: inference() uses
        # them to invert the log-normalisation. Saving weights alone means a
        # later load() silently rescales predictions with whatever statistics
        # the current training split happens to have.
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "y_mean": self.y_mean,
                "y_std": self.y_std,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
            self.y_mean = torch.as_tensor(checkpoint["y_mean"])
            self.y_std = torch.as_tensor(checkpoint["y_std"])
        else:
            # Legacy checkpoint: weights only, normalisation left as set up.
            self.model.load_state_dict(checkpoint)
        self.model.to(self.device)

    def inference(self, item):
        self.model.eval()
        with torch.no_grad():
            vector = self.vectorizer.transform([item.summary])
            vector = torch.FloatTensor(vector.toarray()).to(self.device)
            pred = self.model(vector)[0]
            result = torch.exp(pred * self.y_std + self.y_mean) - 1
            result = result.item()
        return max(0, result)
