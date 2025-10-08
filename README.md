```markdown
# Fashion MNIST Classification

Neural network built from scratch using NumPy to classify fashion items. Same architecture as my MNIST digit recognizer, just applied to clothing instead of numbers.

## What it does

Classifies 28x28 grayscale images into 10 clothing categories (shirts, pants, shoes, bags, etc.).

## Architecture

```
784 (input) -> 10 (hidden, ReLU) -> 10 (output, Softmax)
```

Exact same network as my MNIST project - wanted to see how it performs on a harder classification problem.

## Results

**Accuracy: ~82%** on test data

Lower than MNIST digits (88%) - turns out distinguishing a shirt from a t-shirt is harder than telling apart digits.

## Why this dataset

After MNIST digits, I wanted to try the same architecture on a harder problem. Fashion MNIST uses the same format but the items are less distinct. Got 82% vs 88% on MNIST, which shows that not all classification problems are equal even with the same setup.

## Interesting findings

Confusion matrix shows the network struggles most with:
- Shirt vs T-shirt
- Pullover vs Coat
- Sneaker vs Ankle boot

Makes sense - even humans might confuse these.

## Running it

```bash
git clone https://github.com/SyntaxNomad/fashion-mnist
cd fashion-mnist
pip install -r requirements.txt
python main.py
```

Download `fashion-mnist_train.csv` from Kaggle first.
