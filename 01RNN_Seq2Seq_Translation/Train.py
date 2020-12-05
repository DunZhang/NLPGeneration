from config import Seq2SeqConfig
from torch import optim
import logging
from EncoderDecoder import EncoderRNN, AttnDecoderRNN
from TranslationDataSet import TranslationDataSet
import os
import torch
import torch.nn as nn
import random


def train(conf: Seq2SeqConfig):
    # get data set
    logging.info("get data set")
    data = TranslationDataSet(data_path=conf.data_path, max_length=conf.max_length,
                              sos_token=conf.sos_token, eos_token=conf.eos_token, is_ch=conf.is_ch)
    # get device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(conf.device)
    conf.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # define model
    logging.info("define model and optimizer")
    encoder = EncoderRNN(len(data.src_word2idx), conf.hidden_size).to(conf.device)
    decoder = AttnDecoderRNN(conf.hidden_size, len(data.tar_word2idx), dropout_p=0.1, max_length=conf.max_length).to(
        conf.device)
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=conf.learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=conf.learning_rate)

    criterion = nn.NLLLoss()
    global_step = 1
    total_loss = 0
    for epoch in range(conf.num_epoches):
        for step, (input_tensor, target_tensor) in enumerate(data, start=1):
            global_step += 1
            # 1* seq_len
            input_tensor = input_tensor.to(conf.device)
            # 1 * seq_len
            target_tensor = target_tensor.to(conf.device)

            loss = _train_step(input_tensor, target_tensor, encoder,
                               decoder, encoder_optimizer, decoder_optimizer, criterion, conf)
            total_loss += loss
            if step % conf.print_steps == 0:
                logging.info(
                    "epoch:{},\tstep:{}/{},\tloss:{}".format(epoch, step, data.num_steps_per_epoch,
                                                             total_loss / conf.print_steps))
                total_loss = 0

            if conf.save_steps > 0 and global_step % conf.save_steps == 0:
                logging.info("save to local")
                torch.save(encoder.state_dict(), os.path.join(conf.save_dir, "encoder_{}_{}.bin".format(epoch, step)))
                torch.save(decoder.state_dict(), os.path.join(conf.save_dir, "decoder_{}_{}.bin".format(epoch, step)))
        if conf.save_steps > 0 and global_step % conf.save_steps == 0:
            logging.info("save to local")
            torch.save(encoder.state_dict(), os.path.join(conf.save_dir, "encoder_{}_{}.bin".format(epoch, step)))
            torch.save(decoder.state_dict(), os.path.join(conf.save_dir, "decoder_{}_{}.bin".format(epoch, step)))


def _train_step(input_tensor, target_tensor, encoder: EncoderRNN, decoder: AttnDecoderRNN, encoder_optimizer,
                decoder_optimizer, criterion, conf: Seq2SeqConfig):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    #  target_tensor 1 * seq_len
    target_length = target_tensor.size(1)

    encoder_outputs = torch.zeros(conf.max_length, encoder.hidden_size, device=conf.device)

    loss = 0

    encoder_output, encoder_hidden = encoder(input_tensor)
    encoder_output = encoder_output[0]  # seq_len * hidden_size
    encoder_outputs.index_copy_(dim=0,
                                index=torch.arange(0, encoder_output.shape[0], dtype=torch.long, device=conf.device),
                                source=encoder_output)

    decoder_input = torch.tensor([[conf.sos_token]], device=conf.device)  # 1*1

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < conf.teacher_forcing_ratio else False
    # use_teacher_forcing = False
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[:, di])
            decoder_input = target_tensor[0, di]  # Teacher forcing
            decoder_input = decoder_input.view(1, -1)

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            # encoder_outputs max_len*hidden_size
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[:, di])
            if decoder_input.item() == conf.eos_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / target_length
