class Seq2SeqConfig():
    def __init__(self):
        self.device = 0
        self.hidden_size = 256
        self.learning_rate = 0.01
        self.num_epoches = 50
        self.max_length = 16
        self.sos_token = 0
        self.eos_token = 1
        self.teacher_forcing_ratio = 0.5
        self.print_steps = 5000
        self.save_steps = -1
        self.save_dir = "model/"
        self.data_path = "data/lccc_chat.txt"
        self.is_ch = True
