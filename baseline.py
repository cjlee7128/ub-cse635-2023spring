import argparse 
import json 
from typing import List, Tuple 
from collections import Counter 

from tqdm import tqdm 

import torch 
from torch.utils.data import DataLoader 

from transformers import PreTrainedTokenizer, T5ForConditionalGeneration, T5Tokenizer, AdamW, set_seed 


def parse_command_line_arguments():

    parser = argparse.ArgumentParser(
        description='CLI for training T5 Knowledge-Grounded Dialogue Model')

    parser.add_argument('--t5_model', type=str, default="t5-small",
                        help="What type of T5 model do you want use?")

    parser.add_argument('--batch_size', type=int, default=16,
                        help='mini-batch size (default: 16)')

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

class FaithDial_Dataset(torch.utils.data.Dataset):
    def __init__(self, questions, answers, tokenizer):
        self.tokenizer = tokenizer 
        self.questions = questions 
        self.answers = answers 

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

def train(model: T5ForConditionalGeneration, tokenizer: PreTrainedTokenizer, optimizer: AdamW, train_set: FaithDial_Dataset, validation_set: FaithDial_Dataset, num_train_epochs: int, device: str, batch_size: int, max_input_length: int=512):
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

            # replace padding target token id's of the labels by -100, crossEntropy skip target label == -100
            encoded_targets[encoded_targets == tokenizer.pad_token_id] = -100

            input_ids = input_ids.to(device)
            encoded_targets = encoded_targets.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=encoded_targets)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
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
                model_predictions = model.generate(
                    input_ids=encoded_inputs, attention_mask=attention_mask)

                model_predictions_encoded += model_predictions.tolist()
                target_encoded += encoded_targets.tolist()
        f1, exact_match = validation_set.evaluate(model_predictions_encoded, target_encoded)

        print(f"\t Validation F1 = {f1:.2f}, EM = {exact_match:.2f}")
        if f1 > f1_old :
            model.save_pretrained(f'{args.save_path}/{model.name_or_path}/model/best-f1')
            tokenizer.save_pretrained(f'{args.save_path}/{model.name_or_path}/tokenizer/best-f1')
            f1_old = f1
        if epoch+1 % 10 == 0:
            model.save_pretrained(f'{args.save_path}/{model.name_or_path}/model/checkpoint-{epoch+1}')
            tokenizer.save_pretrained(f'{args.save_path}/{model.name_or_path}/tokenizer/checkpoint-{epoch+1}')
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

    model = T5ForConditionalGeneration.from_pretrained(args.t5_model)
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
            model_predictions = model.generate(
                input_ids=encoded_inputs, attention_mask=attention_mask)

            model_predictions_encoded += model_predictions.tolist()
            target_encoded += encoded_targets.tolist()
    f1, exact_match = test_set.evaluate(model_predictions_encoded, target_encoded)
    print(f"\t Test F1 = {f1:.2f}, EM = {exact_match:.2f}") 