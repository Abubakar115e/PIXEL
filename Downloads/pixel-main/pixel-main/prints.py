from pixel import (
    AutoConfig,
    AutoModelForTokenClassification,
    UPOS_LABELS,
    PIXELSelfAttention,
    PIXELTrainer,
    PIXELTrainingArguments,
    PIXELForTokenClassification,

)
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load a pre-trained model
model = PIXELForTokenClassification.from_pretrained("/mnt/c/Users/abuli/Desktop/Pixel-eval/ViT-Base-Amh").to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

print(f"Number of parameters before pruning ViT MAE base: {count_parameters(model)}")
#Number of parameters before pruning VIT mae base: 85805577
#Number of parameters before pruning PIXEL: 85800194
#Number of parameters before pruning ViT-MAE: 657074508
#Number of parameters after pruning PIXEL UD_Vietnamese-VTB: 21695249
#Number of parameters after pruning ViT-MAE UD_Vietnamese-VTB: 227609097

