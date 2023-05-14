import argparse 
import json 
from typing import List, Tuple 
from collections import Counter 

from tqdm import tqdm 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader 

from transformers import PreTrainedTokenizer, T5ForConditionalGeneration, T5Tokenizer, AdamW, set_seed 

def parse_command_line_arguments():

    parser = argparse.ArgumentParser(
        description='CLI for training T5 Knowledge-Grounded Dialogue Model')

    parser.add_argument('--t5_model', type=str, default="t5-small",
                        help="What type of T5 model do you want use?")

    parser.add_argument('--pos_eps', type=float, default=3.0) 
    parser.add_argument('--neg_eps', type=float, default=1.0) 
    parser.add_argument('--tau', type=float, default=0.1) 

    parser.add_argument('--batch_size', type=int, default=32,
                        help='mini-batch size (default: 32)')

    parser.add_argument('--epochs', type=int, default=40,
                        help='number of training epochs (default: 40)')

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (Adam) (default: 1e-4)')

    parser.add_argument('--workers', type=int, default=10,
                        help='number of working units used to load the data (default: 10)')

    parser.add_argument('--device', default='cuda', type=str,
                        help='device to be used for computations (in {cpu, cuda:0, cuda:1, ...}, default: cpu)')

    parser.add_argument('--max_input_length', type=int, default=512,
                        help='Maximum lenght of input text, (default: 512, maximum admitted: 512)')

    parser.add_argument('--seed', type=int, default=7,
                        help='Seed for random initialization (default: 7)') 

    parser.add_argument('--save_path', type=str, default='results', 
                        help='path to save trained model and tokenizer (default: results)')

    parsed_arguments = parser.parse_args()

    return parsed_arguments 

class AdvContrastiveT5(nn.Module): 
    def __init__(self, t5_model, tau, neg_eps, pos_eps): 
        super(AdvContrastiveT5, self).__init__() 
        self.tau = tau 
        self.neg_eps = neg_eps 
        self.pos_eps = pos_eps 

        self.t5_model = T5ForConditionalGeneration.from_pretrained(t5_model) 
        hidden_size = self.t5_model.config.hidden_size

        self.projection = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU()) 

    def forward(self, input_ids, attention_mask,
                decoder_input_ids, decoder_attention_mask,
                lm_labels, adv=False):
        # input_ids: ids of article tokens
        # attention_mask: mask for input_ids 0 for PAD 1 o.w
        # decoder_input_ids: ids of summary tokens
        # decoder_attention_mask: mask for decoder_input_ids 0 for PAD 1 o.w
        # lm_labels: shift decoder_input_ids left

        encoder = self.t5_model.get_encoder()
        decoder = self.t5_model.get_decoder()

        encoder_outputs = encoder(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  )

        hidden_states = encoder_outputs[0]

        decoder_outputs = decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
        )
        sequence_output = decoder_outputs[0]
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.t5_model.model_dim ** -0.5)
        lm_logits = self.t5_model.lm_head(sequence_output)

        # Add hidden states and attention if they are here
        decoder_outputs = (lm_logits,) + decoder_outputs[1:]

        vocab_size = lm_logits.size(-1)

        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        nll = criterion(lm_logits.view(-1, vocab_size),
                        lm_labels.view(-1))

        if adv:
            proj_enc_h = self.projection(hidden_states)
            proj_dec_h = self.projection(sequence_output)
            avg_doc = self.avg_pool(proj_enc_h, attention_mask)
            avg_abs = self.avg_pool(proj_dec_h, decoder_attention_mask)

            cos = nn.CosineSimilarity(dim=-1)
            cont_crit = nn.CrossEntropyLoss()
            sim_matrix = cos(avg_doc.unsqueeze(1),
                             avg_abs.unsqueeze(0))
            perturbed_dec = self.generate_adv(sequence_output,
                                              lm_labels)  # [n,b,t,d] or [b,t,d]
            batch_size = input_ids.size(0)

            proj_pert_dec_h = self.projection(perturbed_dec)
            avg_pert = self.avg_pool(proj_pert_dec_h,
                                     decoder_attention_mask)

            adv_sim = cos(avg_doc, avg_pert).unsqueeze(1)  # [b,1]

            pos_dec_hidden = self.generate_cont_adv(hidden_states, attention_mask,
                                                    sequence_output, decoder_attention_mask,
                                                    lm_logits,
                                                    self.tau, self.pos_eps)
            avg_pos_dec = self.avg_pool(self.projection(pos_dec_hidden),
                                        decoder_attention_mask)

            pos_sim = cos(avg_doc, avg_pos_dec).unsqueeze(-1)  # [b,1]
            logits = torch.cat([sim_matrix, adv_sim], 1) / self.tau

            identity = torch.eye(batch_size, device=input_ids.device)
            pos_sim = identity * pos_sim
            neg_sim = sim_matrix.masked_fill(identity == 1, 0)
            new_sim_matrix = pos_sim + neg_sim
            new_logits = torch.cat([new_sim_matrix, adv_sim], 1)

            labels = torch.arange(batch_size,
                                  device=input_ids.device)

            cont_loss = cont_crit(logits, labels)
            new_cont_loss = cont_crit(new_logits, labels)

            cont_loss = 0.5 * (cont_loss + new_cont_loss)

            return nll, cont_loss

        else:
            return nll

    def generate_adv(self, dec_hiddens, lm_labels):
        dec_hiddens = dec_hiddens.detach()

        dec_hiddens.requires_grad = True

        lm_logits = self.t5_model.lm_head(dec_hiddens)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        loss = criterion(lm_logits.view(-1, lm_logits.size(-1)),
                         lm_labels.view(-1))

        loss.backward()

        dec_grad = dec_hiddens.grad.detach()
        l2_norm = torch.norm(dec_grad, dim=-1)

        dec_grad /= (l2_norm.unsqueeze(-1) + 1e-12)

        perturbed_dec = dec_hiddens + self.neg_eps * dec_grad.detach()
        perturbed_dec = perturbed_dec  # [b,t,d]
        self.zero_grad()

        return perturbed_dec

    def generate_cont_adv(self, enc_hiddens, enc_mask,
                          dec_hiddens, dec_mask, lm_logits,
                          tau, eps):
        enc_hiddens = enc_hiddens.detach()
        dec_hiddens = dec_hiddens.detach()
        lm_logits = lm_logits.detach()
        dec_hiddens.requires_grad = True

        avg_enc = self.avg_pool(self.projection(enc_hiddens),
                                enc_mask)

        avg_dec = self.avg_pool(self.projection(dec_hiddens),
                                dec_mask)

        cos = nn.CosineSimilarity(dim=-1)
        logits = cos(avg_enc.unsqueeze(1), avg_dec.unsqueeze(0)) / tau

        cont_crit = nn.CrossEntropyLoss()
        labels = torch.arange(avg_enc.size(0),
                              device=enc_hiddens.device)
        loss = cont_crit(logits, labels)
        loss.backward()

        dec_grad = dec_hiddens.grad.detach()
        l2_norm = torch.norm(dec_grad, dim=-1)
        dec_grad /= (l2_norm.unsqueeze(-1) + 1e-12)

        perturb_dec_hidden = dec_hiddens + eps * dec_grad
        perturb_dec_hidden = perturb_dec_hidden.detach()
        perturb_dec_hidden.requires_grad = True
        perturb_logits = self.t5_model.lm_head(perturb_dec_hidden)

        true_probs = F.softmax(lm_logits, -1)
        true_probs = true_probs * dec_mask.unsqueeze(-1).float()

        perturb_log_probs = F.log_softmax(perturb_logits, -1)

        kl_crit = nn.KLDivLoss(reduction="sum")
        vocab_size = lm_logits.size(-1)

        kl = kl_crit(perturb_log_probs.view(-1, vocab_size),
                     true_probs.view(-1, vocab_size))
        kl = kl / torch.sum(dec_mask).float()
        kl.backward()

        kl_grad = perturb_dec_hidden.grad.detach()

        l2_norm = torch.norm(kl_grad, dim=-1)

        kl_grad /= (l2_norm.unsqueeze(-1) + 1e-12)

        perturb_dec_hidden = perturb_dec_hidden - eps * kl_grad
        # self.zero_grad()

        return perturb_dec_hidden

    def avg_pool(self, hidden_states, mask):
        length = torch.sum(mask, 1, keepdim=True).float()
        mask = mask.unsqueeze(2)
        hidden = hidden_states.masked_fill(mask == 0, 0.0)
        avg_hidden = torch.sum(hidden, 1) / length

        return avg_hidden

class FaithDial_Dataset(torch.utils.data.Dataset):
    def __init__(self, questions, answers, tokenizer):
        self.tokenizer = tokenizer 
        self.questions = questions 
        self.answers = answers 

        # https://github.com/nunziati/bert-vs-t5-for-question-answering/blob/main/MyDataset.py 
        if len(self.questions) != len(self.answers):
            raise Exception(
                "something wrong while building the dataset: questions and answers result in different dimensions")

        self.item_count: int = len(self.questions)

    def __getitem__(self, index: int):
        return self.questions[index], self.answers[index]

    def __len__(self):
        return self.item_count 

    def pack_minibatch(self, data: List[Tuple[str, str]]):
        """Pack mini-batch function
        """
        return zip(*data) 

    def __exact_match_score(self, prediction, ground_truth):
        """_summary_
        Args:
            prediction (_type_): _description_
            ground_truth (_type_): _description_
        Returns:
            _type_: _description_
        """
        if len(ground_truth) == len(prediction):
            if all(token1 == token2 for token1, token2 in zip(ground_truth,prediction)):
                return 1
        return 0 

    def __f1_score(self, prediction_tokens, ground_truth_tokens):
        """_summary_
        Args:
            prediction (_type_): _description_
            ground_truth (_type_): _description_
        Returns:
            _type_: _description_
        """
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1 

    def evaluate(self, predictions, gold_answers):
        """_summary_
        Args:
            predictions (_type_): _description_
            gold_answers (_type_): _description_
        Returns:
            _type_: _description_
        """
        f1 = exact_match = 0

        for ground_truths, prediction in tqdm(zip(gold_answers, predictions)):
            # Remove pad token
            tokens_to_remove = {
                self.tokenizer.pad_token_id,
                self.tokenizer.eos_token_id,
                self.tokenizer.bos_token_id,
                self.tokenizer.cls_token_id,
                self.tokenizer.sep_token_id,
                self.tokenizer.mask_token_id
            }
            prediction = list(filter(lambda token: token not in tokens_to_remove, prediction))
            ground_truths = list(filter(lambda token: token not in tokens_to_remove, ground_truths))
            f1 += self.__f1_score(prediction, ground_truths)
            exact_match += self.__exact_match_score(prediction, ground_truths) 
        return 100*f1/len(predictions), 100*exact_match/len(predictions) 

def train(model, tokenizer: PreTrainedTokenizer, optimizer: AdamW, train_set: FaithDial_Dataset, validation_set: FaithDial_Dataset, num_train_epochs: int, device: str, batch_size: int, max_input_length: int=512):
    """_summary_
    Args:
        model (T5ForConditionalGeneration): _description_
        tokenizer (PreTrainedTokenizer): _description_
        optimizer (AdamW): _description_
        train_set (Dataset): _description_
        validation_set (Dataset): _description_
        num_train_epochs (int): _description_
        device (str): _description_
        batch_size (int): _description_
    """
    my_trainset_dataloader = DataLoader(train_set, batch_size=args.batch_size,
                                        num_workers=args.workers, collate_fn=lambda data: train_set.pack_minibatch(data))
    my_validation_dataloader = DataLoader(validation_set, batch_size=args.batch_size,
                                          num_workers=args.workers, collate_fn=lambda data: validation_set.pack_minibatch(data))

    model.zero_grad() 
    # set training mode on the model
    model.train()

    # model to device
    model.to(device)

    f1_old: int = 0
    for epoch in range(num_train_epochs):
        epoch_train_loss = 0.
        for questions, answers in tqdm(my_trainset_dataloader):
            optimizer.zero_grad()

            inputs = questions
            encoded_inputs = tokenizer(
                                    inputs,
                                    padding="longest",
                                    max_length=max_input_length,
                                    truncation=True,
                                    return_tensors="pt",
                                )
            encoded_targets = tokenizer(
                                    answers,
                                    padding="longest",
                                    max_length=max_input_length,
                                    truncation=True,
                                    return_tensors="pt",
                                )

            input_ids, attention_mask = encoded_inputs.input_ids, encoded_inputs.attention_mask
            encoded_targets = encoded_targets.input_ids
            # decoder_input_ids must not have negative index value(s). 
            decoder_input_ids = model.t5_model._shift_right(encoded_targets) 
            encoded_targets[encoded_targets == tokenizer.pad_token_id] = -100 

            decoder_attention_mask = torch.sign(decoder_input_ids) 
            decoder_attention_mask[:, 0] = 1 
            decoder_attention_mask = decoder_attention_mask.to(device) 

            input_ids = input_ids.to(device)
            encoded_targets = encoded_targets.to(device)
            attention_mask = attention_mask.to(device)
            decoder_input_ids = decoder_input_ids.to(device)

            nll, cont_loss = model(input_ids=input_ids, attention_mask=attention_mask, 
                                   decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, 
                                   lm_labels=encoded_targets, adv=True) 
            loss = nll + cont_loss 
            loss.backward()
            optimizer.step()
            model.zero_grad() 
            epoch_train_loss += loss.item() * batch_size
        print(f"epoch={epoch + 1}/{num_train_epochs}")
        print(f"\t Train loss = {epoch_train_loss/len(train_set):.4f}")

        model.eval()
        with torch.no_grad():
            model_predictions_encoded = []
            target_encoded = []
            for questions, answers in tqdm(my_validation_dataloader):
                
                inputs = questions 
                encoded_inputs = tokenizer(
                    inputs,
                    padding="longest",
                    max_length=max_input_length,
                    truncation=True,
                    return_tensors="pt",
                )
                encoded_targets = tokenizer(
                    answers,
                    padding="longest",
                    max_length=max_input_length,
                    truncation=True,
                    return_tensors="pt",
                )
                encoded_inputs, attention_mask = encoded_inputs.input_ids, encoded_inputs.attention_mask
                encoded_targets = encoded_targets.input_ids

                encoded_inputs = encoded_inputs.to(device)
                encoded_targets = encoded_targets.to(device)
                attention_mask = attention_mask.to(device)
                model_predictions = model.t5_model.generate(
                    input_ids=encoded_inputs, attention_mask=attention_mask)

                model_predictions_encoded += model_predictions.tolist()
                target_encoded += encoded_targets.tolist()
        f1, exact_match = validation_set.evaluate(model_predictions_encoded, target_encoded)

        print(f"\t Validation F1 = {f1:.2f}, EM = {exact_match:.2f}")
        if f1 > f1_old :
            model.t5_model.save_pretrained(f'{args.save_path}/{args.t5_model}/model/best-f1') 
            tokenizer.save_pretrained(f'{args.save_path}/{args.t5_model}/tokenizer/best-f1')
            f1_old = f1
        if epoch+1 % 10 == 0:
            model.t5_model.save_pretrained(f'{args.save_path}/{args.t5_model}/model/checkpoint-{epoch+1}')
            tokenizer.save_pretrained(f'{args.save_path}/{args.t5_model}/tokenizer/checkpoint-{epoch+1}')
        model.train()

def build_data(data_path, hal_data_path): 

    question_list = ['Is the response hallucinated?', 'What are the response attribution classes?', 'What are the speech acts?', 'What is the faithful response to this?'] 
    
    questions = [] 
    answers = [] 

    with open(data_path, 'r') as f1, open(hal_data_path, 'r') as f2: 
        data = json.load(f1) 
        hal_data = json.load(f2) 

    for conversation, hal_conversation in zip(data, hal_data): 
        for turn, hal_turn in zip(conversation['utterances'], hal_conversation['utterances']): 
            # Task1 
            # question, history, knowledge, response -> "Yes" or "No" 
            # knowledge and response are usually short 
            questions.append(f"question: {question_list[0]} knowledge: {turn['knowledge']} response: {turn['response']} history: {', '.join(turn['history'])}")
            answers.append("No") 

            hal_turn['history'] = ['null' if h is None else h for h in hal_turn['history']]
            hal_turn['response'] = 'null' if hal_turn['response'] is None else hal_turn['response'] 
            # knowledge and response are usually short 
            questions.append(f"question: {question_list[0]} knowledge: {hal_turn['knowledge']} response: {hal_turn['response']} history: {', '.join(hal_turn['history'])}") 
            answers.append("Yes")

            # Task2-1 
            # question, history, knowledge, response -> BEGIN tag(s) 
            # knowledge and response are usually short 
            questions.append(f"question: {question_list[1]} knowledge: {hal_turn['knowledge']} response: {hal_turn['response']} history: {', '.join(hal_turn['history'])}") 
            answers.append(f"{', '.join(hal_turn['BEGIN'])}") 

            # Task2-2 
            # question, history, knowledge -> VRM tag(s) 
            if hal_turn['VRM'][0] != '': 
                # knowledge is usually short 
                questions.append(f"question: {question_list[2]} knowledge: {hal_turn['knowledge']} history: {', '.join(hal_turn['history'])}") 
                answers.append(f"{', '.join(['Acknowledgment' if v=='Ack.' else v for v in hal_turn['VRM']])}") 

            # Task3 
            # question, history, knowledge -> response 
            # knowledge is usually short 
            questions.append(f"question: {question_list[3]} knowledge: {turn['knowledge']} history: {', '.join(turn['history'])}")
            answers.append(f"{turn['response']}") 

    return questions, answers 

if __name__ == '__main__': 
    args = parse_command_line_arguments() 
    set_seed(args.seed) 

    train_questions, train_answers = build_data('data/train.json', 'data/hal_train.json') 
    val_questions, val_answers = build_data('data/valid.json', 'data/hal_valid.json') 
    test_questions, test_answers = build_data('data/test.json', 'data/hal_test.json') 

    print(len(train_questions), len(train_answers))
    print(len(val_questions), len(val_answers))
    print(len(test_questions), len(test_answers))

    model = AdvContrastiveT5(args.t5_model, args.tau, args.neg_eps, args.pos_eps) 
    tokenizer = T5Tokenizer.from_pretrained(args.t5_model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_set = FaithDial_Dataset(train_questions, train_answers, tokenizer) 
    val_set = FaithDial_Dataset(val_questions, val_answers, tokenizer) 
    test_set = FaithDial_Dataset(test_questions, test_answers, tokenizer) 

    train(model=model,
          tokenizer=tokenizer,
          optimizer=optimizer,
          train_set=train_set,
          validation_set=val_set,
          num_train_epochs=args.epochs, device=args.device, batch_size=args.batch_size)
    
    my_test_dataloader = DataLoader(test_set, batch_size=args.batch_size,
                                          num_workers=args.workers, collate_fn=lambda data: test_set.pack_minibatch(data))

    model.eval()
    with torch.no_grad():
        model_predictions_encoded = []
        target_encoded = []
        for questions, answers in tqdm(my_test_dataloader): 
            inputs = questions 
            encoded_inputs = tokenizer(
                inputs,
                padding="longest",
                max_length=args.max_input_length,
                truncation=True,
                return_tensors="pt",
            )
            encoded_targets = tokenizer(
                answers,
                padding="longest",
                max_length=args.max_input_length,
                truncation=True,
                return_tensors="pt",
            )
            encoded_inputs, attention_mask = encoded_inputs.input_ids, encoded_inputs.attention_mask
            encoded_targets = encoded_targets.input_ids

            encoded_inputs = encoded_inputs.to(args.device)
            encoded_targets = encoded_targets.to(args.device)
            attention_mask = attention_mask.to(args.device)
            model_predictions = model.t5_model.generate(
                input_ids=encoded_inputs, attention_mask=attention_mask)

            model_predictions_encoded += model_predictions.tolist()
            target_encoded += encoded_targets.tolist()
    f1, exact_match = test_set.evaluate(model_predictions_encoded, target_encoded)
    print(f"\t Test F1 = {f1:.2f}, EM = {exact_match:.2f}") 