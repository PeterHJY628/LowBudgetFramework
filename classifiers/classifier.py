from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from core.data import BaseDataset
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from classifiers.seeded_layers import SeededLinear
from classifiers.lstm import BiLSTMModel


class DINOv3Classifier(nn.Module):
    def __init__(self,
                 model_rng,
                 num_classes:int,
                 model_name:str,
                 freeze_backbone:bool=True,
                 input_size:int=224,
                 cifar_mean:tuple=(0.4914, 0.4822, 0.4465),
                 cifar_std:tuple=(0.2023, 0.1994, 0.2010),
                 imagenet_mean:tuple=(0.485, 0.456, 0.406),
                 imagenet_std:tuple=(0.229, 0.224, 0.225)):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as e:
            raise ImportError("Using classifier.type='DINOv3' requires the 'transformers' package") from e

        self.backbone = AutoModel.from_pretrained(model_name)
        self.freeze_backbone = freeze_backbone
        self.input_size = input_size
        self.register_buffer("cifar_mean", torch.tensor(cifar_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("cifar_std", torch.tensor(cifar_std).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("imagenet_mean", torch.tensor(imagenet_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("imagenet_std", torch.tensor(imagenet_std).view(1, 3, 1, 1), persistent=False)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.hidden_size = self.backbone.config.hidden_size
        self.head = SeededLinear(model_rng, self.hidden_size, num_classes)
        self._features_cached = False

    def _preprocess(self, x:Tensor)->Tensor:
        x = x * self.cifar_std + self.cifar_mean
        x = torch.clamp(x, 0.0, 1.0)
        x = (x - self.imagenet_mean) / self.imagenet_std
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bicubic", align_corners=False)
        return x

    def _encode(self, x:Tensor)->Tensor:
        if self._features_cached:
            return x
        x = self._preprocess(x)
        if self.freeze_backbone:
            with torch.no_grad():
                out = self.backbone(pixel_values=x)
        else:
            out = self.backbone(pixel_values=x)

        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            return out.last_hidden_state[:, 0]
        if isinstance(out, tuple) and len(out) > 0:
            return out[0][:, 0]
        raise ValueError("Unsupported DINOv3 output format")

    def extract_features(self, x:Tensor, batch_size:int=64)->Tensor:
        """One-time bulk feature extraction through the frozen backbone."""
        assert self.freeze_backbone, "Feature caching only supported with frozen backbone"
        all_features = []
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                batch = x[start:start + batch_size]
                features = self._encode(batch)
                all_features.append(features)
        return torch.cat(all_features, dim=0)

    def enable_feature_cache(self):
        """Switch to cached mode: forward skips backbone, expects pre-extracted features."""
        self._features_cached = True
        self.backbone.cpu()
        self.backbone = None
        torch.cuda.empty_cache()

    def forward(self, x:Tensor)->Tensor:
        features = self._encode(x)
        return self.head(features)


class LinearModel(nn.Module):
    def __init__(self, model_rng, input_size:int, num_classes:int, dropout=None):
        super().__init__()
        self.dropout = dropout
        self.out = SeededLinear(model_rng, input_size, num_classes)

    def _encode(self, x:Tensor)->Tensor:
        return x

    def forward(self, x:Tensor)->Tensor:
        if self.dropout is not None:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.out(x)
        return x


class DenseModel(nn.Module):
    def __init__(self, model_rng, input_size:int, num_classes:int, hidden_sizes:tuple, dropout=None, add_head=True):
        assert len(hidden_sizes) > 0
        super().__init__()
        self.dropout = dropout

        self.inpt = SeededLinear(model_rng, input_size, hidden_sizes[0])
        self.hidden = nn.ModuleList()
        for i in range(len(hidden_sizes)):
            self.hidden.append(SeededLinear(model_rng, hidden_sizes[max(0, i - 1)], hidden_sizes[i]))
        if add_head:
            self.out = SeededLinear(model_rng, hidden_sizes[-1], num_classes)

    def _encode(self, x:Tensor)->Tensor:
        """
        The split bewteen encoding and prediction is important for agents that use latent features from the
        classifier like Coreset
        """
        if len(x.size()) == 4:
            # Pretext SimCLR workaround
            x = x.squeeze()
        x = self.inpt(x)
        x = F.relu(x)
        if self.dropout is not None:
            x = F.dropout(x, self.dropout, training=self.training)
        for h_layer in self.hidden:
            x = h_layer(x)
            x = F.relu(x)
            if self.dropout is not None:
                x = F.dropout(x, self.dropout, training=self.training)
        return x

    def forward(self, x:Tensor)->Tensor:
        x = self._encode(x)
        if hasattr(self, "out"):
            x = self.out(x)
        return x


class ConvolutionalModel(nn.Module):
    def __init__(self, input_size:Tuple[int], num_classes:int, hidden_sizes:Tuple[int]):
        assert len(hidden_sizes) > 0
        assert len(input_size) > 1 and len(input_size) < 4
        if len(input_size) == 2:
            print("found greyscale input. adding a color dimension for compatibility")
            input_size = (1, *input_size)
        super().__init__()

        self.inpt = nn.Conv2d(input_size[0], hidden_sizes[0], kernel_size=3)
        self.hidden = nn.ModuleList()
        for i in range(len(hidden_sizes)):
            self.hidden.append(nn.Conv2d(hidden_sizes[max(0, i - 1)], hidden_sizes[i], kernel_size=3))
        self.flatten = nn.Flatten()

        test_inpt = torch.zeros((1, *input_size))
        test_out = self._encode(test_inpt)

        self.out = nn.Linear(test_out.shape[-1], num_classes)

    def _encode(self, x:Tensor)->Tensor:
        """
        The split bewteen encoding and prediction is important for agents that use latent features from the
        classifier like Coreset
        """
        x = self.inpt(x)
        x = F.relu(x)
        for h_layer in self.hidden:
            x = h_layer(x)
            x = F.relu(x)
        x = self.flatten(x)
        return x

    def forward(self, x:Tensor)->Tensor:
        x = self._encode(x)
        x = self.out(x)
        return x


def construct_model(model_rng, dataset:BaseDataset, model_config:dict, add_head=True) -> Tuple[nn.Module, int]:
        '''
        Constructs the model by name and additional parameters
        Returns model and its output dim
        '''
        x_shape = dataset.x_shape
        n_classes = dataset.n_classes
        model_type = model_config["type"].lower()
        dropout = model_config["dropout"] if "dropout" in model_config else None
        if model_type == "linear":
            return LinearModel(model_rng, x_shape[-1], n_classes, dropout), \
                   n_classes
        elif model_type == "resnet18":
            from classifiers.resnet import ResNet18, load_pretrained_backbone, freeze_backbone
            model = ResNet18(model_rng=model_rng,
                             num_classes=n_classes, in_channels=x_shape[0],
                             dropout=dropout,
                             add_head=add_head)
            if "pretrained_path" in model_config and model_config["pretrained_path"]:
                load_pretrained_backbone(model, model_config["pretrained_path"])
            if model_config.get("freeze_backbone", False):
                freeze_backbone(model)
            return model, n_classes if add_head else 512
        elif model_type == "dinov3":
            if not add_head:
                raise ValueError("DINOv3 is only supported as classifier with a head (add_head=True)")
            model = DINOv3Classifier(
                model_rng=model_rng,
                num_classes=n_classes,
                model_name=model_config.get("model_name", "facebook/dinov3-vitb16-pretrain-lvd1689m"),
                freeze_backbone=model_config.get("freeze_backbone", True),
                input_size=model_config.get("input_size", 224),
            )
            return model, n_classes
        elif model_type == "mlp":
            return DenseModel(model_rng,
                              input_size=x_shape[-1],
                              num_classes=n_classes,
                              hidden_sizes=model_config["hidden"],
                              dropout=dropout,
                              add_head=add_head), \
                   n_classes if add_head else model_config["hidden"][-1]
        elif model_type == "bilstm":
            assert hasattr(dataset, "embedding_data_file"), "Dataset is missing the embedding file. This is specific to text datasets."
            embedding_data = torch.load(dataset.embedding_data_file)
            return BiLSTMModel(model_rng,
                               embedding_data=embedding_data,
                               num_classes=n_classes,
                               dropout=dropout), \
                   n_classes if add_head else model_config["hidden"][-1]
        else:
            raise NotImplementedError



def fit_and_evaluate(dataset:BaseDataset,
                     model_rng,
                     disable_progess_bar:bool=False,
                     max_epochs:int=4000,
                     patience:int=40):

    from core.helper_functions import EarlyStopping
    loss = nn.CrossEntropyLoss()
    model = dataset.get_classifier(model_rng)
    model = model.to(dataset.device)
    # optimizer = dataset.get_optimizer(model)
    if dataset.encoded:
        optim_cfg = dataset.config["optimizer_embedded"]
    else:
        optim_cfg = dataset.config["optimizer"]
    optimizer = dataset.get_optimizer(model)
    # optimizer = torch.optim.Adam(model.parameters(), lr=optim_cfg["lr"], weight_decay=optim_cfg["weight_decay"])

    train_dataloader = DataLoader(TensorDataset(dataset.x_train, dataset.y_train),
                                  batch_size=dataset.classifier_batch_size,
                                  shuffle=True)
    val_dataloader = DataLoader(TensorDataset(dataset.x_val, dataset.y_val), batch_size=512)
    test_dataloader = DataLoader(TensorDataset(dataset.x_test, dataset.y_test), batch_size=512)
    all_accs = []
    early_stop = EarlyStopping(patience=patience)
    iterator = tqdm(range(max_epochs), disable=disable_progess_bar, miniters=2)
    for e in iterator:
        model.train()
        for batch_x, batch_y in train_dataloader:
            yHat = model(batch_x)
            loss_value = loss(yHat, batch_y)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        # early stopping on validation
        model.eval()
        with torch.no_grad():
            loss_sum = 0.0
            for batch_x, batch_y in val_dataloader:
                yHat = model(batch_x)
                class_loss = loss(yHat, torch.argmax(batch_y.long(), dim=1))
                loss_sum += class_loss.detach().cpu().numpy()
            if early_stop.check_stop(loss_sum):
                print(f"Early stop after {e} epochs")
                break

        correct = 0.0
        test_loss = 0.0
        model.eval()
        for batch_x, batch_y in test_dataloader:
            yHat = model(batch_x)
            predicted = torch.argmax(yHat, dim=1)
            correct += (predicted == torch.argmax(batch_y, dim=1)).sum().item()
            class_loss = loss(yHat, torch.argmax(batch_y.long(), dim=1))
            test_loss += class_loss.detach().cpu().numpy()
        test_acc = correct / len(dataset.x_test)
        all_accs.append(test_acc)
        iterator.set_postfix({"test loss": "%1.4f"%test_loss, "test acc": "%1.4f"%test_acc})
    return all_accs


if __name__ == '__main__':
    import yaml
    import numpy as np
    from core.helper_functions import get_dataset_by_name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = "splice"
    with open(f"configs/{dataset}.yaml", 'r') as f:
        config = yaml.load(f, yaml.Loader)
    DatasetClass = get_dataset_by_name(dataset)
    DatasetClass.inject_config(config)
    pool_rng = np.random.default_rng(1)
    dataset = DatasetClass("../datasets", config, pool_rng, encoded=0)
    dataset = dataset.to(device)
    model_rng = torch.Generator()
    model_rng.manual_seed(1)
    accs = fit_and_evaluate(dataset, model_rng)
    import matplotlib.pyplot as plt
    plt.plot(accs)

