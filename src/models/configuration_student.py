from transformers import PretrainedConfig

class StudentConfig(PretrainedConfig):
    def __init__(
        self,
        base_model='gpt2',
        tokenizer_name='gpt2',
        mixture_size=1,
        np = True,
        np_input_dim = None,
        np_num_hidden = None,
        np_num_latent = 1024,
        np_use_cross_attn = True,
        np_use_transformer = True,
        np_t_nhead = 8,
        np_t_num_lyrs = 3,
        np_t_dim_feedforward = 1024,
        np_t_dropout = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.tokenizer_name = tokenizer_name
        self.mixture_size = mixture_size

        self.np = np
        self.np_input_dim = np_input_dim
        self.np_num_hidden = np_num_hidden
        self.np_num_latent = np_num_latent
        self.np_use_cross_attn = np_use_cross_attn
        self.np_use_transformer = np_use_transformer
        self.np_t_nhead = np_t_nhead
        self.np_t_num_lyrs = np_t_num_lyrs
        self.np_t_dim_feedforward = np_t_dim_feedforward
        self.np_t_dropout = np_t_dropout

