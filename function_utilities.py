import json
from transformers import BertTokenizerFast
from transformers import BertForQuestionAnswering
from torch.utils.data import DataLoader
import torch
import config

def create_encoding(tokenizer_name, contexts, questions, answers,
                    truncation = True, padding = True):
    #global tokenizer
    config.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
    input_ids = config.tokenizer(contexts, questions, truncation = truncation, padding=padding)
    encodings = add_token_positions(input_ids, answers, config.tokenizer)
    return encodings

def instantiate_BERT_model(problem_type = "question_answering",
                           model_name = "bert-base-uncased"):
    #global model
    if problem_type == "question_answering":
        config.model = BertForQuestionAnswering.from_pretrained(model_name)
        return config.model
    else:
        raise NotImplementedError

def add_token_positions(input_ids, answers, tokenizer):
    start_positions = []
    end_positions = []

    for i in range(len(answers)):
        start_positions.append(input_ids.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(input_ids.char_to_token(i, answers[i]['answer_end'] - 1))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length

    input_ids.update({'start_positions': start_positions, 'end_positions': end_positions})
    return input_ids

def add_end_idx(answers, contexts):

    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        # sometimes squad answers are off by a character or two so we fix this
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        elif context[start_idx-1:end_idx-1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
        elif context[start_idx-2:end_idx-2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters

    return answers, contexts

def read_data(path):
    # load the json file
    with open(path, 'rb') as f:
        squad = json.load(f)

    contexts = []
    questions = []
    answers = []

    for group in squad['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)

    return contexts, questions, answers

def create_dataloader(dataset, batch_size = 16, shuffle = True):
    loader = DataLoader(dataset, batch_size, shuffle)
    return loader

def save_artifacts(trained_model, tokenizer, artifact_directory):

    trained_model.save_pretrained(artifact_directory)
    tokenizer.save_pretrained(artifact_directory)

    print('all artifacts are saved')


def get_prediction(context, question, model, tokenizer, device):
    inputs = tokenizer.encode_plus(question, context, return_tensors='pt').to(device)
    outputs = model(**inputs)

    answer_start = torch.argmax(outputs[0])
    answer_end = torch.argmax(outputs[1]) + 1

    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))

    return answer


def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(prediction, truth):
    return bool(normalize_text(prediction) == normalize_text(truth))


def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()

    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)

    return round(2 * (prec * rec) / (prec + rec), 2)


def question_answer(context, question, answer):
    prediction = get_prediction(context, question, config.model, config.tokenizer, config.device)
    em_score = exact_match(prediction, answer)
    f1_score = compute_f1(prediction, answer)

    print(f'Question: {question}')
    print(f'Prediction: {prediction}')
    print(f'True Answer: {answer}')
    print(f'Exact match: {em_score}')
    print(f'F1 score: {f1_score}\n')

