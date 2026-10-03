# Data

The datasets are **not included** in this repository because of their size and licences. `Capstone_Phase3A_Paper_Replication.ipynb` downloads them automatically into `data/raw/` on first run.

| Dataset | Used as | Source | Licence / terms |
|---|---|---|---|
| GoEmotions (Demszky et al., 2020) | 31,621 Reddit comments with one Ekman label | Google Research raw rater-level release: `https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_{1,2,3}.csv` | See the GoEmotions repository |
| EmotionLines — Friends (Chen et al., 2018) | 11,731 utterances (`non-neutral` removed) | EmotionX 2019 release: `https://github.com/bshmueli/EmotionX-2019` (`2019_Train_Friends.zip`) | CC BY-NC-ND 4.0 |
| Twitter Emotion Corpus (Mohammad, 2012) | 21,051 tweets, 6 emotions | `http://saifmohammad.com/WebDocs/Jan9-2012-tweets-clean.txt.zip` | Research use; see the author's site |

Derived files (`data/processed/`: LM scores, sentence embeddings, grid predictions) are created by the notebooks and are also not committed.
