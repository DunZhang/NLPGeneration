import torch
import random
from TranslationDataSet import TranslationDataSet
import os
from EncoderDecoder import EncoderRNN, AttnDecoderRNN
from collections import OrderedDict


def evaluate(encoder, decoder, sentence, src_word2idx: OrderedDict, tar_idx2word: OrderedDict, device: torch.device,
             max_length=10, sos_token=0, eos_token=1):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        input_tensor = torch.LongTensor([[src_word2idx[word] for word in sentence.split(" ")] + [eos_token]]).to(device)
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        encoder_output, encoder_hidden = encoder(input_tensor)
        encoder_output = encoder_output[0]  # seq_len * hidden_size
        encoder_outputs.index_copy_(dim=0,
                                    index=torch.arange(0, encoder_output.shape[0], dtype=torch.long, device=device),
                                    source=encoder_output)

        decoder_input = torch.tensor([[sos_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == eos_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(tar_idx2word[topi.item()])

            decoder_input = topi.detach()

        return decoded_words, decoder_attentions[:di + 1]


#######################################################################################################################
def evaluateRandomly(data: TranslationDataSet, encoder, decoder, device: torch.device, n=10):
    for i in range(n):
        pair = random.choice(data.pairs)
        print('src sentence:\t', pair[0])
        print('tar sentence:\t', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0], data.src_word2idx, data.tar_idx2word, device)
        output_sentence = ' '.join(output_words)
        print('pred sentence:\t', output_sentence)
        print('')


if __name__ == "__main__":
    device = 0
    hidden_size = 256
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = TranslationDataSet("data/eng-fra.txt")
    encoder = EncoderRNN(len(data.src_word2idx), hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, len(data.tar_word2idx), dropout_p=0.1)

    encoder.load_state_dict(torch.load("model/encoder_5_7004.bin", map_location="cpu"))
    encoder = encoder.to(device)

    decoder.load_state_dict(torch.load("model/decoder_5_7004.bin", map_location="cpu"))
    decoder = decoder.to(device)

    evaluateRandomly(data, encoder, decoder, device)
