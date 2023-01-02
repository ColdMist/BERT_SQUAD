import os
from class_utilities import SQuAD_Dataset
from function_utilities import (read_data, add_end_idx, create_encoding,
                                instantiate_BERT_model, config, create_dataloader,
                                save_artifacts
                                )
from train_eval_utilities import do_train, do_eval



if __name__ == '__main__':
    '''
    Implementation follows https://github.com/angelosps/Question-Answering/blob/main/Question_Answering.ipynb
    '''
    data_dir = './data/BERT-SQuAD'
    artifact_directory = os.path.join(data_dir, 'model_artifacts')
    # Load the training dataset and take a look at it
    # with open('./data/BERT-SQuAD/train-v2.0.json', 'rb') as f:
    #   squad = json.load(f)
    #find group about greece

    # Find the group about Greece
    # gr = -1
    # for idx, group in enumerate(squad['data']):
    #   print(group['title'])
    #   if group['title'] == 'Greece':
    #     gr = idx
    #     print(gr)
    #     break
    #
    # and this is the context given for NYC
    #squad['data'][186]['paragraphs'][0]['context']

    #Read the questions and answerpairs from raw data, NB the data is not formatted!

    train_contexts, train_questions, train_answers = read_data(os.path.join(data_dir,'train-v2.0.json'))
    valid_contexts, valid_questions, valid_answers = read_data(os.path.join(data_dir,'dev-v2.0.json'))

    train_answers, train_contexts = add_end_idx(train_answers, train_contexts)
    valid_answers, valid_contexts = add_end_idx(valid_answers, valid_contexts)
    #
    #
    # # You can see that now we get the answer_end also
    # print(train_questions[-10000])
    # print(train_answers[-10000])
    #
    train_encodings = create_encoding(tokenizer_name = 'bert-base-uncased', contexts=train_contexts,
                                      questions=train_questions, answers=train_answers, truncation=True, padding=True)
    valid_encodings = create_encoding(tokenizer_name='bert-base-uncased', contexts=valid_contexts,
                                      questions=valid_questions, answers=valid_answers, truncation=True, padding=True)
    #tokenizer_name = 'bert-base-uncased'
    #tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)

    #train_input_ids = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
    #valid_input_ids = tokenizer(valid_contexts, valid_questions, truncation=True, padding=True)

    #no_of_encodings = len(train_input_ids['input_ids'])
    #print(f'We have {no_of_encodings} context-question pairs')

    #train_encodings = add_token_positions(train_input_ids, train_answers)
    #valid_encodings = add_token_positions(valid_input_ids, valid_answers)

    #Define the dataloaders
    #print(config.tokenizer)
    #exit()
    train_dataset = SQuAD_Dataset(train_encodings)
    valid_dataset = SQuAD_Dataset(valid_encodings)

    #instantiate BERT for question answering model
    config.model = instantiate_BERT_model()

    # Check on the available device - use GPU
    #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    train_loader = create_dataloader(train_dataset, batch_size=1, shuffle=True)
    valid_loader = create_dataloader(valid_dataset, batch_size=1, shuffle=False)
    #
    trained_model = do_train(config.model, config.device, train_loader, epochs=5)

    save_artifacts(trained_model, config.tokenizer, artifact_directory)

    do_eval(config.model, config.device, valid_loader)
