import json
import torch
import timm

def extract_model_config(model):
    """Extracts relevant configuration parameters from a pruned Vision Transformer model."""
    config = {
        "attention_probs_dropout_prob": model.default_cfg.get('attention_probs_dropout_prob', 0.0),
        "hidden_size": model.embed_dim,
        "image_size": model.default_cfg['input_size'][-1],
        "initializer_range": 0.02,  # Default value used in many transformers
        "intermediate_size": model.blocks[0].mlp.fc1.in_features if hasattr(model.blocks[0].mlp.fc1, 'in_features') else None,
        "layer_norm_eps": model.norm_eps,
        "num_attention_heads": model.num_heads,
        "num_channels": model.default_cfg['input_size'][0],
        "num_hidden_layers": len(model.blocks),
        "patch_size": model.patch_embed.patch_size[0] if hasattr(model.patch_embed, 'patch_size') else None,
        "qkv_bias": model.qkv.bias is not None,
        "hidden_act": "gelu",  # Assuming GELU is the activation function
        "hidden_dropout_prob": model.drop,
        "torch_dtype": str(model.default_cfg.get('dtype', torch.float32)),
        "transformers_version": "4.16.0.dev0"  # Example version
    }
    return config

def main():
    model_path = 'pruned_vit_cifar10.pth'
    model_name = 'vit_huge_patch14_224'  # Replace with the model architecture you used

    # Load the pruned model
    model = timm.create_model(model_name, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Extract configuration
    config = extract_model_config(model)

    # Save the configuration to a JSON file
    config_path = 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"Configuration saved to {config_path}")

if __name__ == '__main__':
    main()
