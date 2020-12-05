from Train import train
from config import Seq2SeqConfig
import logging
import torch

logging.basicConfig(level=logging.INFO)
if __name__ == "__main__":
    conf = Seq2SeqConfig()
    train(conf)
