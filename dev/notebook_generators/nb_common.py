# Shared cells for the Phase 3B notebook generator (setup, data, features, graph, model, post-processing, evaluation).
# Imported by make_improved_nb.py, which adds the experiment cells.

SETUP = r'''
import os, sys, re, json, html, random, time, itertools, warnings, subprocess
from collections import Counter, defaultdict

IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torch_geometric'], check=True)
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT_DIR = '/content/drive/MyDrive/Capstone Project'   # change if the folder is elsewhere in Drive
else:
    PROJECT_DIR = os.getcwd()

PROC_DIR = os.path.join(PROJECT_DIR, 'data', 'processed')
REP_DIR = os.path.join(PROJECT_DIR, 'results', 'replication')
RES_DIR = os.path.join(PROJECT_DIR, 'results', 'improved')
RUN_DIR = os.path.join(RES_DIR, 'runs')            # cached predictions of every trained model (lets re-runs resume)
for d in (RES_DIR, RUN_DIR):
    os.makedirs(d, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import scipy.sparse as sp
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter
from transformers import AutoTokenizer, AutoModel
from transformers.utils import logging as hf_logging
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
hf_logging.set_verbosity_error()
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Project dir:', PROJECT_DIR)
print('Device     :', DEVICE, torch.cuda.get_device_name(0) if DEVICE == 'cuda' else '')
'''
