
# Pruning PIXEL

Run the following shell scripts to prune the model for POS run ./pos.sh and for NER run ./ner.sh.
For the VGG model use the VGG-pruner.ipynb notebook and  ViT-MAE for run cifar-pruner.py. the difference files for run_pos.py and run_ner.py are provided in the script/training folder.

## Configurations to use ViT-MAE models

You need to change the following configurations in the config.json and text_renderer_config.json files to use the ViT-MAE models. If task is NER then use the following image size "image_size": [16, 3136] for both config.json and text_renderer_config.json files. If task is POS then use the following image size "image_size": [16, 4096] for both config.json and text_renderer_config.json files.

## Results

The results of the pruned models and non pruned models are stored in the Pixel-eval folder. The results are stored in the form of single merged txt file and the subfolders for each model. The subfolders contain the results of each model in the form of json files, where they are stored in the form of all_results.json, config.json and the contain the training, evalution and testing results of the model.