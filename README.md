# Deep Learning Projects

Projects I built during my Deep Learning course this semester. Two of them
use backprop written by hand in NumPy, no autodiff.

| Project | What it does | Built with |
|---|---|---|
| [Digit to speech](01-digits-to-speech) | MNIST digit image in, spoken-digit waveform out (2,983 samples). 5-layer MLP, hand-written backprop | NumPy, MATLAB |
| [Autoencoder with few labels](02-autoencoder-few-labels) | Does autoencoder pretraining help when only 10% of images are labeled? Jaguar vs Tiger | PyTorch |
| [RNN sequence completion](03-rnn-sequence-completion) | Recurrent net learns a hidden rule from 20 sequences. Hand-written BPTT, gradient check, LOO CV, 100% test | NumPy |
| [Tang poem GPT](04-tang-poem-gpt) | Trained a small character-level GPT on Tang poems on CPU, generated a poem, translated it to Bengali | PyTorch ([tangshi-gpt](https://github.com/dengfeng-ai/tangshi-gpt)) |

Datasets and trained weights aren't included because they're too big. Each
project's README links to the data and shows how to run it.

```bash
pip install -r requirements.txt
```
