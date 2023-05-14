import os 
import argparse 
import json 
from typing import List, Tuple 
from collections import Counter 

from tqdm import tqdm 

import torch 
from torch.utils.data import DataLoader 

from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, set_seed 
from transformers import DistilBertTokenizer, DistilBertModel

# https://stackoverflow.com/questions/64156202/add-dense-layer-on-top-of-huggingface-bert-model 
class CustomDistilBertModel(torch.nn.Module): 
    def __init__(self, bert_model_name, t5_seq_len, t5_emb_dim): 
        super(CustomDistilBertModel, self).__init__() 
        self.t5_seq_len = t5_seq_len 
        self.t5_emb_dim = t5_emb_dim 
        self.db_model = DistilBertModel.from_pretrained(bert_model_name) 
        # (B, bert_seq_len, 768) -> (B, t5_seq_len, t5_emb_dim) 
        self.linear1 = torch.nn.Linear(768, t5_emb_dim) 

    def forward(self, input_ids, attention_mask): 
        # (B, bert_seq_len, 768) 
        db_outputs = self.db_model(input_ids, attention_mask=attention_mask).last_hidden_state 
        # (B, t5_seq_len, t5_emb_dim) 
        if db_outputs.size(1) < self.t5_seq_len: 
            diff = self.t5_seq_len - db_outputs.size(1) 
            db_outputs = torch.cat((db_outputs, db_outputs[:, -1:, :].repeat(1, diff, 1)), 1) 
        linear1_output = self.linear1(db_outputs[:, :self.t5_seq_len, :]) 

        return linear1_output 

def parse_command_line_arguments():

    parser = argparse.ArgumentParser(
        description='CLI for training T5 Knowledge-Grounded Dialogue Model')

    parser.add_argument('--t5_model', type=str, default="t5-small",
                        help="What type of T5 model do you want use?") 
    
    parser.add_argument('--bert_model', type=str, default="distilbert-base-uncased", 
                        help="What type of BERT model do you want use for encoding?") 

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

    parser.add_argument('--max_input_length', type=int, default=32,
                        help='Maximum length of input text for T5 model, (default: 32, maximum admitted: 512)')
    
    parser.add_argument('--bert_max_input_length', type=int, default=512,
                        help='Maximum length of input text for bert encoder, (default: 512, maximum admitted: 512)')

    parser.add_argument('--seed', type=int, default=7,
                        help='Seed for random initialization (default: 7)') 

    parser.add_argument('--save_path', type=str, default='results', 
                        help='path to save trained model and tokenizer (default: results)')

    parsed_arguments = parser.parse_args()

    return parsed_arguments

class FaithDial_Dataset(torch.utils.data.Dataset):
    def __init__(self, questions, contexts, answers, tokenizer):
        self.tokenizer = tokenizer 
        self.questions = questions 
        self.contexts = contexts 
        self.answers = answers 

        # https://github.com/nunziati/bert-vs-t5-for-question-answering/blob/main/MyDataset.py 
        if len(self.questions) != len(self.contexts): 
            raise Exception(
                "something wrong while building the dataset: questions and contexts in different dimensions") 
        if len(self.questions) != len(self.answers):
            raise Exception(
                "something wrong while building the dataset: questions and answers result in different dimensions")

        self.item_count: int = len(self.questions)

    def __getitem__(self, index: int):
        return self.questions[index], self.contexts[index], self.answers[index]

    def __len__(self):
        return self.item_count 

    def pack_minibatch(self, data: List[Tuple[str, str, str]]):
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

def train(models, tokenizers, optimizer: AdamW, train_set: FaithDial_Dataset, validation_set: FaithDial_Dataset, num_train_epochs: int, device: str, batch_size: int, max_input_length: int=64, bert_max_input_length: int=512):
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

    t5_model = models[0] 
    db_model = models[1] 

    # set training mode on the model
    t5_model.train() 
    db_model.train() 

    # model to device
    t5_model.to(device) 
    db_model.to(device) 

    t5_tokenizer = tokenizers[0] 
    db_tokenizer = tokenizers[1] 

    f1_old: int = 0
    for epoch in range(num_train_epochs):
        epoch_train_loss = 0.
        for questions, contexts, answers in tqdm(my_trainset_dataloader):
            optimizer.zero_grad()

            db_inputs = contexts 
            db_encoded_inputs = db_tokenizer(db_inputs, 
                                             padding="longest", 
                                             max_length=bert_max_input_length, 
                                             truncation=True, 
                                             return_tensors="pt").to(device)
            db_outputs = db_model(**db_encoded_inputs) 

            inputs = questions 
            encoded_inputs = t5_tokenizer(
                                    inputs,
                                    # padding="longest",
                                    padding='max_length', 
                                    max_length=max_input_length,
                                    truncation=True,
                                    return_tensors="pt",
                                )
            encoded_targets = t5_tokenizer(
                                    answers,
                                    padding="longest",
                                    max_length=max_input_length,
                                    truncation=True,
                                    return_tensors="pt",
                                )

            input_ids, attention_mask = encoded_inputs.input_ids, encoded_inputs.attention_mask
            encoded_targets = encoded_targets.input_ids
            # decoder_input_ids must not have negative index value(s). 
            decoder_input_ids = t5_model._shift_right(encoded_targets) 
            encoded_targets[encoded_targets == t5_tokenizer.pad_token_id] = -100 

            decoder_attention_mask = torch.sign(decoder_input_ids) 
            decoder_attention_mask[:, 0] = 1 
            decoder_attention_mask = decoder_attention_mask.to(device) 

            input_ids = input_ids.to(device)
            encoded_targets = encoded_targets.to(device)
            attention_mask = attention_mask.to(device)
            decoder_input_ids = decoder_input_ids.to(device)

            t5_encoder = t5_model.get_encoder() 
            t5_decoder = t5_model.get_decoder() 

            t5_encoder_outputs = t5_encoder(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            ) 
            hidden_states = t5_encoder_outputs[0] 

            t5_decoder_outputs = t5_decoder(input_ids=decoder_input_ids,
                                            attention_mask=decoder_attention_mask,
                                            encoder_hidden_states=(hidden_states + db_outputs) / 2,
                                            encoder_attention_mask=attention_mask,
                                            )
            sequence_output = t5_decoder_outputs[0] 
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (t5_model.model_dim ** -0.5)
            lm_logits = t5_model.lm_head(sequence_output) 

            vocab_size = lm_logits.size(-1)

            criterion = torch.nn.CrossEntropyLoss(ignore_index=-100) 
            loss = criterion(lm_logits.view(-1, vocab_size),
                             encoded_targets.view(-1)) 
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * batch_size
        print(f"epoch={epoch + 1}/{num_train_epochs}")
        print(f"\t Train loss = {epoch_train_loss/len(train_set):.4f}")

        t5_model.eval()
        db_model.eval() 
        with torch.no_grad():
            model_predictions_encoded = []
            target_encoded = []
            for questions, contexts, answers in tqdm(my_validation_dataloader):
                
                db_inputs = contexts 
                db_encoded_inputs = db_tokenizer(db_inputs, 
                                                padding="longest", 
                                                max_length=bert_max_input_length, 
                                                truncation=True, 
                                                return_tensors="pt").to(device)
                db_outputs = db_model(**db_encoded_inputs) 

                inputs = questions 
                encoded_inputs = t5_tokenizer(
                    inputs,
                    # padding="longest", 
                    padding='max_length', 
                    max_length=max_input_length,
                    truncation=True,
                    return_tensors="pt",
                )
                encoded_targets = t5_tokenizer(
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

                t5_encoder = t5_model.get_encoder() 

                t5_encoder_outputs = t5_encoder(input_ids=encoded_inputs,
                                                attention_mask=attention_mask,
                                                ) 
                t5_encoder_outputs['last_hidden_state'] = (t5_encoder_outputs['last_hidden_state'] + db_outputs) / 2 

                model_predictions = t5_model.generate(input_ids=encoded_inputs,
                                                      attention_mask=attention_mask,
                                                      encoder_outputs=t5_encoder_outputs, 
                                                      )

                model_predictions_encoded += model_predictions.tolist()
                target_encoded += encoded_targets.tolist()
        f1, exact_match = validation_set.evaluate(model_predictions_encoded, target_encoded)

        print(f"\t Validation F1 = {f1:.2f}, EM = {exact_match:.2f}")
        if f1 > f1_old :
            t5_model.save_pretrained(f'{args.save_path}/{t5_model.name_or_path}/model/best-f1')
            t5_tokenizer.save_pretrained(f'{args.save_path}/{t5_model.name_or_path}/tokenizer/best-f1')
            torch.save(db_model, f'{args.save_path}/{args.bert_model}/model/best-f1/pytorch_model.pt') 
            db_tokenizer.save_pretrained(f'{args.save_path}/{args.bert_model}/tokenizer/best-f1') 
            f1_old = f1
        if epoch+1 % 10 == 0:
            t5_model.save_pretrained(f'{args.save_path}/{t5_model.name_or_path}/model/checkpoint-{epoch+1}')
            t5_tokenizer.save_pretrained(f'{args.save_path}/{t5_model.name_or_path}/tokenizer/checkpoint-{epoch+1}') 
            if not os.path.isdir(f'{args.save_path}/{args.bert_model}/model/checkpoint-{epoch+1}'): 
                os.makedirs(f'{args.save_path}/{args.bert_model}/model/checkpoint-{epoch+1}') 
            torch.save(db_model, f'{args.save_path}/{args.bert_model}/model/checkpoint-{epoch+1}/pytorch_model.pt') 
            db_tokenizer.save_pretrained(f'{args.save_path}/{args.bert_model}/tokenizer/checkpoint-{epoch+1}')
        t5_model.train()
        db_model.train() 

def build_data(data_path, hal_data_path): 

    question_list = ['Is the response hallucinated?', 'What are the response attribution classes?', 'What are the speech acts?', 'What is the faithful response to this?'] 
    
    questions = [] 
    contexts = [] 
    answers = [] 

    with open(data_path, 'r') as f1, open(hal_data_path, 'r') as f2: 
        data = json.load(f1) 
        hal_data = json.load(f2) 

    for conversation, hal_conversation in zip(data, hal_data): 
        for turn, hal_turn in zip(conversation['utterances'], hal_conversation['utterances']): 
            # Task1 
            # question, history, knowledge, response -> "Yes" or "No" 
            # knowledge and response are usually short 
            questions.append(f"question: {question_list[0]} response: {turn['response']}")
            contexts.append(f"{turn['knowledge']} {', '.join(turn['history'])}")
            answers.append("No") 

            hal_turn['history'] = ['null' if h is None else h for h in hal_turn['history']]
            hal_turn['response'] = 'null' if hal_turn['response'] is None else hal_turn['response'] 
            # knowledge and response are usually short 
            questions.append(f"question: {question_list[0]} response: {hal_turn['response']}") 
            contexts.append(f"{hal_turn['knowledge']} {', '.join(hal_turn['history'])}") 
            answers.append("Yes")

            # Task2-1 
            # question, history, knowledge, response -> BEGIN tag(s) 
            # knowledge and response are usually short 
            questions.append(f"question: {question_list[1]} response: {hal_turn['response']}") 
            contexts.append(f"{hal_turn['knowledge']} {', '.join(hal_turn['history'])}") 
            answers.append(f"{', '.join(hal_turn['BEGIN'])}") 

            # Task2-2 
            # question, history, knowledge -> VRM tag(s) 
            if hal_turn['VRM'][0] != '': 
                # knowledge is usually short 
                questions.append(f"question: {question_list[2]}") 
                contexts.append(f"{hal_turn['knowledge']} {', '.join(hal_turn['history'])}") 
                answers.append(f"{', '.join(['Acknowledgment' if v=='Ack.' else v for v in hal_turn['VRM']])}") 

            # Task3 
            # question, history, knowledge -> response 
            # knowledge is usually short 
            questions.append(f"question: {question_list[3]}") 
            contexts.append(f"{turn['knowledge']} {', '.join(turn['history'])}") 
            answers.append(f"{turn['response']}") 

    return questions, contexts, answers 

if __name__ == '__main__': 
    args = parse_command_line_arguments() 
    set_seed(args.seed) 

    train_questions, train_contexts, train_answers = build_data('data/train.json', 'data/hal_train.json') 
    val_questions, val_contexts, val_answers = build_data('data/valid.json', 'data/hal_valid.json') 
    test_questions, test_contexts, test_answers = build_data('data/test.json', 'data/hal_test.json') 

    print(len(train_questions), len(train_contexts), len(train_answers))
    print(len(val_questions), len(val_contexts), len(val_answers))
    print(len(test_questions), len(test_contexts), len(test_answers))

    t5_model = T5ForConditionalGeneration.from_pretrained(args.t5_model)
    t5_tokenizer = T5Tokenizer.from_pretrained(args.t5_model)

    db_tokenizer = DistilBertTokenizer.from_pretrained(args.bert_model) 
    db_model = CustomDistilBertModel(args.bert_model, t5_seq_len=args.max_input_length, 
                                     t5_emb_dim=512) 
    
    if not os.path.isdir(f'{args.save_path}/{args.bert_model}/model/best-f1'): 
        os.makedirs(f'{args.save_path}/{args.bert_model}/model/best-f1') 
    
    optimizer = torch.optim.AdamW(list(t5_model.parameters()) + list(db_model.parameters()), lr=args.lr) 

    train_set = FaithDial_Dataset(train_questions, train_contexts, train_answers, t5_tokenizer) 
    val_set = FaithDial_Dataset(val_questions, val_contexts, val_answers, t5_tokenizer) 
    test_set = FaithDial_Dataset(test_questions, test_contexts, test_answers, t5_tokenizer) 

    train(models=[t5_model, db_model],
          tokenizers=[t5_tokenizer, db_tokenizer],
          optimizer=optimizer,
          train_set=train_set,
          validation_set=val_set,
          num_train_epochs=args.epochs, device=args.device, batch_size=args.batch_size, 
          max_input_length=args.max_input_length, bert_max_input_length=args.bert_max_input_length) 
    
    my_test_dataloader = DataLoader(test_set, batch_size=args.batch_size,
                                    num_workers=args.workers, collate_fn=lambda data: test_set.pack_minibatch(data))

    t5_model.eval()
    db_model.eval() 
    with torch.no_grad():
        model_predictions_encoded = []
        target_encoded = []
        for questions, contexts, answers in tqdm(my_test_dataloader): 
            db_inputs = contexts 
            db_encoded_inputs = db_tokenizer(db_inputs, 
                                            padding="longest", 
                                            max_length=args.bert_max_input_length, 
                                            truncation=True, 
                                            return_tensors="pt").to(args.device)
            db_outputs = db_model(**db_encoded_inputs) 

            inputs = questions 
            encoded_inputs = t5_tokenizer(
                inputs,
                padding='max_length', 
                max_length=args.max_input_length,
                truncation=True,
                return_tensors="pt",
            )
            encoded_targets = t5_tokenizer(
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

            t5_encoder = t5_model.get_encoder() 

            t5_encoder_outputs = t5_encoder(input_ids=encoded_inputs,
                                            attention_mask=attention_mask,
                                            ) 
            t5_encoder_outputs['last_hidden_state'] = (t5_encoder_outputs['last_hidden_state'] + db_outputs) / 2 

            model_predictions = t5_model.generate(input_ids=encoded_inputs,
                                                    attention_mask=attention_mask,
                                                    encoder_outputs=t5_encoder_outputs, 
                                                    )

            model_predictions_encoded += model_predictions.tolist()
            target_encoded += encoded_targets.tolist()
    f1, exact_match = test_set.evaluate(model_predictions_encoded, target_encoded)
    print(f"\t Test F1 = {f1:.2f}, EM = {exact_match:.2f}") 