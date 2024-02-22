import sys
import os
import logging
import torch
import torch.nn as nn
from transformers import AutoTokenizer

sys.path.append("..")
from utils import get_sep_position
from .configuration_student import StudentConfig
from .modeling_gpt2_implicit import GPT2LMHeadImplicitModel
from .neural_process import *

class Student(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.base_model = GPT2LMHeadImplicitModel.from_pretrained(config.base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        num_layers = len(self.base_model.transformer.h)
        hidden_size = self.base_model.config.hidden_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size


        
        self.latent_encoder = None
        self.deterministic_encoder = None
        self.decoder = None
        if hasattr(config, 'np'):
            if config.np:
                self.latent_encoder = LatentEncoder(input_dim=config.np_input_dim, num_hidden=config.np_num_hidden,
                                                    num_latent=config.np_num_latent, cross_attn=config.np_use_cross_attn,
                                                    transformer=config.np_use_transformer, t_nhead=config.np_t_nhead,
                                                    t_num_lyrs=config.np_t_num_lyrs, t_dim_feedforward=config.np_t_dim_feedforward,
                                                    t_dropout=config.np_t_dropout)
                self.deterministic_encoder = DeterministicEncoder(num_hidden=config.np_num_hidden, input_dim=config.np_input_dim,
                                                                  transformer=config.np_use_transformer, t_nhead=config.np_t_nhead,
                                                                  t_num_lyrs=config.np_t_num_lyrs, t_dim_feedforward=config.np_t_dim_feedforward,
                                                                  t_dropout=config.np_t_dropout)
                self.decoder = Decoder(output_dim=config.np_input_dim, num_hidden=config.np_num_hidden, num_lyrs=config.np_t_num_lyrs)

        self.mlps = nn.ModuleList([nn.Sequential(
                 nn.Linear(hidden_size, 4*hidden_size),
                 nn.ReLU(),
                 nn.Linear(4*hidden_size, hidden_size),
                 ) for _ in range(num_layers)])
        
        self.starter_mlps = nn.ModuleList([nn.Sequential(
                 nn.Linear(hidden_size, 4*hidden_size),
                 nn.ReLU(),
                 nn.Linear(4*hidden_size, hidden_size),
                 ) for _ in range(3)])

    def forward(self, input_ids, positions_to_substitute, teacher_states, output_hidden_states=False):
        outputs = self.base_model.forward(mode='forward_student', \
                input_ids=input_ids, \
                positions_to_substitute=positions_to_substitute, \
                states_to_substitute=teacher_states, \
                output_hidden_states=output_hidden_states, \
                mlps=self.starter_mlps,
                np=self.config.np,
                np_latent_encoder=self.latent_encoder,
                np_deterministic_encoder=self.deterministic_encoder,
                np_decoder=self.decoder)
        return outputs

    def compute_loss(self, input_ids, labels, teacher_states):
        #import pdb; pdb.set_trace()
        sep_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id)
        # First, project teacher states
        teacher_states = [self.mlps[l](teacher_states[l]) for l in range(len(teacher_states))]
        batch_size = input_ids.shape[0]

        # Forward while substituting teacher states
        outputs = self.forward(input_ids, sep_positions, teacher_states)
        logits = outputs.logits
        np_states = outputs.np_states

        labels_pred = logits.argmax(-1)
        mask = labels[...,1:].ge(0)
        correct_tokens = ((labels_pred[...,:-1] == labels[...,1:]) * mask).sum()
        total_tokens = mask.sum()
        token_accuracy = correct_tokens / total_tokens

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss()
        loss1 = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        loss_fct = nn.MSELoss(reduction='none')
        loss2 = 0
        for teacher_state, student_state in zip(teacher_states, outputs.f_h_cs):
            loss2 += loss_fct(teacher_state, student_state).sum(-1) / 2
        loss2 = loss2.mean()

        kl = 0
        for np_state in np_states:
            prior_mu, prior_var, posterior_mu, posterior_var = np_state
            kl += self.kl_div(prior_mu, prior_var, posterior_mu, posterior_var)
        kl = kl.mean()

        outputs.loss = loss1 + loss2 + kl
        outputs.token_accuracy = token_accuracy
        outputs.total_correct = correct_tokens
        outputs.total_loss = loss1 * total_tokens + loss2 * batch_size + kl * len(np_states)
        outputs.total_tokens = total_tokens
        return outputs

    def generate(self, input_ids, teacher_states, max_new_tokens=512, num_beams=1):
        sep_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id)
        batch_size = input_ids.shape[0]
        beam_output = []
        # First, project teacher states
        teacher_states = [self.mlps[l](teacher_states[l]) for l in range(len(teacher_states))]
        for i in range(batch_size):
            input_ids_i = input_ids[i:i+1]
            sep_positions_i = sep_positions[i:i+1]
            input_ids_i = input_ids_i[:, :sep_positions_i+1]
            beam_output_i = self.base_model.generate(
                input_ids=input_ids_i,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
                num_return_sequences=1,
                positions_to_substitute=sep_positions_i.repeat_interleave(num_beams, dim=0),
                states_to_substitute=[z[i:i+1].repeat_interleave(num_beams, dim=0) for z in teacher_states],
                mode='forward_student',
            )
            beam_output.append(beam_output_i)
        return beam_output
    
    def std_train(self):
        self.base_model.transformer.std_training = True
    
    def std_eval(self):
        self.base_model.transformer.std_training = False

    @classmethod
    def from_pretrained(self, pretrained_path):
        config = StudentConfig.from_pretrained(pretrained_path)
        model = Student(config)
        state_dict = torch.load(os.path.join(pretrained_path, 'state_dict.bin'))
        try:
            model.load_state_dict(state_dict)
        except:
            model.load_state_dict(state_dict, strict=False)
            logging.warn("Some weights of the model Student checkpoint not loaded.")

        return model

    def save_pretrained(self, save_directory):
        print (f'Saving to {save_directory}')
        self.config.save_pretrained(save_directory)
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_directory, 'state_dict.bin'))


def kl_div(self, prior_mu, prior_var, posterior_mu, posterior_var):
        kl_div = (torch.exp(posterior_var) + (posterior_mu-prior_mu) ** 2) / torch.exp(prior_var) - 1. + (prior_var - posterior_var)
        kl_div = 0.5 * kl_div.sum()
        return kl_div

