# Training data

The training corpus is intentionally supplied by the user and is not committed to the repository.

Create a UTF-8 text file and pass its path to `train.py`:

```bash
mkdir -p data
nano data/train.txt
python3 train.py --data data/train.txt
```

For a quick smoke test, the corpus must contain at least `sequence_length + 1` encoded tokens. The default sequence length is 32 bytes/tokens.

Training artifacts are written under `checkpoints/` by default and should remain local. They are not part of the source repository.
