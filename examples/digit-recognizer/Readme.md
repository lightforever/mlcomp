[Kaggle digit-recognizer competition](https://www.kaggle.com/c/digit-recognizer)

Install [Kaggle API](https://github.com/Kaggle/kaggle-api).

Ensure you filled your key in  ~/.kaggle/kaggle.json 

```bash
cd examples/digit-recognizer
```

Download dataset and split on folds

```bash
mlcomp dag download.yml
```

Distributed training. Please fill your gpu count

```bash
mlcomp dag train-distr.yml
```

Distributed training with stages.

```bash
mlcomp dag train-distr-stage.yml
```

Grid search

```bash
mlcomp dag grid.yml
```

All together with an upload of the submission to Kaggle!

```bash
mlcomp dag all.yml
```

You can check your [submissions page](https://www.kaggle.com/c/digit-recognizer/submissions)