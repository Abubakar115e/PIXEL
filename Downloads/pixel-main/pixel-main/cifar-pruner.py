from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import (CenterCrop, Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, Resize, ToTensor)
import pytorch_lightning as pl
from transformers import ViTImageProcessor, ViTForImageClassification, AdamW
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch_pruning as tp
from transformers.models.vit.modeling_vit import ViTSelfAttention, ViTSelfOutput
import os

# Load CIFAR10 (only small portion for demonstration purposes)
train_ds, test_ds = load_dataset('cifar10', split=['train[:5000]', 'test[:2000]'])
splits = train_ds.train_test_split(test_size=0.1)
train_ds = splits['train']
val_ds = splits['test']

id2label = {id: label for id, label in enumerate(train_ds.features['label'].names)}
label2id = {label: id for id, label in id2label.items()}

# Preprocessing the data
processor = ViTImageProcessor.from_pretrained("facebook/vit-mae-huge")
image_mean = processor.image_mean
image_std = processor.image_std
size = processor.size["height"]

normalize = Normalize(mean=image_mean, std=image_std)
_train_transforms = Compose([
    RandomResizedCrop(size),
    RandomHorizontalFlip(),
    ToTensor(),
    normalize,
])

_val_transforms = Compose([
    Resize(size),
    CenterCrop(size),
    ToTensor(),
    normalize,
])

def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['img']]
    return examples

def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['img']]
    return examples

train_ds.set_transform(train_transforms)
val_ds.set_transform(val_transforms)
test_ds.set_transform(val_transforms)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

train_batch_size = 4
eval_batch_size = 4
num_workers = 2  # Set the number of workers to 5

train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=collate_fn, batch_size=train_batch_size, num_workers=num_workers)
val_dataloader = DataLoader(val_ds, collate_fn=collate_fn, batch_size=eval_batch_size, num_workers=num_workers)
test_dataloader = DataLoader(test_ds, collate_fn=collate_fn, batch_size=eval_batch_size, num_workers=num_workers)

class ViTLightningModule(pl.LightningModule):
    def __init__(self, num_labels=10, mask_ratio=0.75):
        super(ViTLightningModule, self).__init__()
        self.vit = ViTForImageClassification.from_pretrained('facebook/vit-mae-huge',
                                                              num_labels=num_labels,
                                                              id2label=id2label,
                                                              label2id=label2id)
        self.mask_ratio = mask_ratio  # Set the mask ratio for masked autoencoding

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits

    def mask_input(self, pixel_values):
        batch_size, channels, height, width = pixel_values.shape
        num_patches = height * width
        num_masked = int(self.mask_ratio * num_patches)

        mask_indices = torch.randperm(num_patches)[:num_masked]
        mask = torch.ones(num_patches)
        mask[mask_indices] = 0
        mask = mask.view(1, 1, height, width).expand(batch_size, channels, -1, -1)

        masked_pixel_values = pixel_values * mask.to(pixel_values.device)
        return masked_pixel_values

    def common_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        masked_pixel_values = self.mask_input(pixel_values)
        logits = self(masked_pixel_values)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        predictions = logits.argmax(-1)
        correct = (predictions == labels).sum().item()
        accuracy = correct / pixel_values.shape[0]
        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)
        self.log("training_loss", loss)
        self.log("training_accuracy", accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss, on_epoch=True)
        self.log("validation_accuracy", accuracy, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=5e-5)

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader

    def test_dataloader(self):
        return test_dataloader

# Pruning technique
device = 'cuda' if torch.cuda.is_available() else 'cpu'
example_inputs = torch.randn(1, 3, 224, 224).to(device)
imp = tp.importance.MagnitudeImportance(p=1)

model = ViTForImageClassification.from_pretrained('facebook/vit-mae-huge', num_labels=10, id2label=id2label, label2id=label2id).to(device)
base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)

num_heads = {}
ignored_layers = [model.classifier]
for m in model.modules():
    if isinstance(m, ViTSelfAttention):
        num_heads[m.query] = m.num_attention_heads
        num_heads[m.key] = m.num_attention_heads
        num_heads[m.value] = m.num_attention_heads
    if isinstance(m, ViTSelfOutput):
        ignored_layers.append(m.dense)

pruner = tp.pruner.MetaPruner(
    model,
    example_inputs,
    global_pruning=False,
    importance=imp,
    pruning_ratio=0.6,
    ignored_layers=ignored_layers,
    output_transform=lambda out: out.logits.sum(),
    num_heads=num_heads,
    prune_head_dims=True,
    prune_num_heads=False,
    head_pruning_ratio=0.5,
)

for g in pruner.step(interactive=True):
    g.prune()

for m in model.modules():
    if isinstance(m, ViTSelfAttention):
        m.num_attention_heads = pruner.num_heads[m.query]
        m.attention_head_size = m.query.out_features // m.num_attention_heads
        m.all_head_size = m.query.out_features

# Integrate the pruned model with the LightningModule
class PrunedViTLightningModule(ViTLightningModule):
    def __init__(self, mask_ratio=0.75):
        super(PrunedViTLightningModule, self).__init__(num_labels=10, mask_ratio=mask_ratio)
        self.vit = model

model = PrunedViTLightningModule()

# Callbacks
early_stop_callback = EarlyStopping(monitor='validation_loss', patience=3, verbose=False, mode='min')
checkpoint_callback = ModelCheckpoint(monitor='validation_loss', save_top_k=1, mode='min', filename='best-checkpoint')

trainer = Trainer(devices='auto', accelerator='auto', precision=16, callbacks=[early_stop_callback, checkpoint_callback], max_epochs=5)
trainer.fit(model)
trainer.test(ckpt_path='best')

# Save the pruned model
pruned_model_path = "pruned_vit_model.pth"
torch.save(model.state_dict(), pruned_model_path)
print(f"Saving the pruned model to {pruned_model_path}...")
