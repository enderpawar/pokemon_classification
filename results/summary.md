| Tag | Backbone | Mode | Epochs | Best Val Acc | Test Acc | Top-5 | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|---|---|---|---|---|
| exp1_resnet18_frozen | resnet18 | feature_extract | 15 | 0.7507 | 0.7713 | 0.9326 | 0.7900 | 0.7822 | 0.7664 |
| exp2_resnet18_full | resnet18 | full_finetune | 15 | 0.9550 | 0.9492 | 0.9892 | 0.9510 | 0.9522 | 0.9478 |
| exp3_resnet50_full | resnet50 | full_finetune | 15 | 0.9560 | 0.9638 | 0.9951 | 0.9622 | 0.9639 | 0.9609 |
| exp4_resnet18_scratch | resnet18 | from_scratch | 15 | 0.6794 | 0.6872 | 0.9091 | 0.6949 | 0.6922 | 0.6698 |
| exp5_mobilenetv2_full | mobilenet_v2 | full_finetune | 15 | 0.9472 | 0.9462 | 0.9922 | 0.9509 | 0.9500 | 0.9452 |
