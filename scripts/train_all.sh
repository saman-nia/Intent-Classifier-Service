set -e

# 1) Train baseline
hatch run python trains/train_baseline.py \
  --train data/atis/train.tsv \
  --test  data/atis/test.tsv  \
  --out-dir models/fasttext   \
  --epochs 20 --embed-dim 128

# 2) Fine-tune BERT
hatch run python trains/train_bert.py \
  --train data/atis/train.tsv \
  --test  data/atis/test.tsv  \
  --out-dir models/bert       \
  --epochs 3
