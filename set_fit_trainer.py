from datasets import load_dataset, Dataset
from sentence_transformers.losses import CosineSimilarityLoss
import pandas as pd
from setfit import SetFitModel, SetFitTrainer, sample_dataset
import json
import numpy as np

with open('./data/data_eval_banking77_30/train.json', 'r') as file:
    train_data = json.load(file)['rasa_nlu_data']

with open('./data/data_eval_banking77_30/test.json', 'r') as file:
    test_data = json.load(file)['rasa_nlu_data']

data_train, label_train = [i['text'] for i in train_data['common_examples']], [str(i['intent']) for i in train_data['common_examples']]
data_test, label_test = [i['text'] for i in test_data['common_examples']], [str(i['intent']) for i in test_data['common_examples']]
# random_index = np.random.choice(np.arange(0,len(data)), max_lim ,replace=False)
# print(random_index)

intent_df_train = pd.DataFrame(np.c_[data_train, label_train], columns=['text', 'label_text'])
intent_df_test = pd.DataFrame(np.c_[data_test, label_test], columns=['text', 'label_text'])

intent_df_train['label'] = intent_df_train['label_text'].astype(int)
intent_df_test['label'] = intent_df_test['label_text'].astype(int)
# dataset = load_dataset("SetFit/SentEval-CR")


# Load SetFit model from Hub
model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")

# train_ds = dataset["train"].shuffle(seed=42).select(range(8 * 2))
# test_ds = dataset["test"]

train_ds = Dataset.from_pandas(intent_df_train)
test_ds = Dataset.from_pandas(intent_df_test)
#print(pd.DataFrame(test_ds).head())
#exit()
# Create trainer
trainer = SetFitTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    loss_class=CosineSimilarityLoss,
    batch_size=16,
    num_iterations=20, # Number of text pairs to generate for contrastive learning
    num_epochs=1, # Number of epochs to use for contrastive learning,
    column_mapping={"text": "text", "label": "label"},

)

# Train and evaluate!
trainer.train(max_length = 128)
metrics = trainer.evaluate()
print(metrics)
# trainer.model.save_pretrained('./pretrained_models/model_test')
#
# model_loaded = SetFitModel.from_pretrained("./pretrained_models/model_test", local_files_only=True)
# x = model_loaded.predict_proba(["I need to activate my card.", "asdasd"])
# x = model_loaded(["i loved the spiderman movie!"])
# print(type(x[0]))
# maximum = np.max(x, axis=-1)
# print(type(maximum))
#preds = model_loaded(["i loved the spiderman movie!", "pineapple on pizza is the worst 🤮"])
#print(preds)

