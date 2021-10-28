from transformers import BertTokenizer, BertForMaskedLM
import torch
import pandas as pd
from transformers import TrainingArguments
from transformers import Trainer

EPOCHS = 50
MODEL_PATH = "models"
TEST_NAME = f"all_question_{EPOCHS}epoch"

df = pd.read_csv("shuffled_qa_consolidated.csv")
print(df.shape)

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-multilingual-uncased')

questions = list(df.loc[:, "question"])
# questions = questions[:20]
question_inputs = tokenizer(questions, return_tensors='pt', max_length=240, 
        truncation=True, padding='max_length')

# create labels
question_inputs["labels"] = question_inputs.input_ids.detach().clone()
# create random array of floats with equal dimensions to input_ids tensor
rand = torch.rand(question_inputs.input_ids.shape)
# create mask array
mask_arr = (rand < 0.15) * (question_inputs.input_ids != 101) * \
           (question_inputs.input_ids != 102) * (question_inputs.input_ids != 0)

selection = []

for i in range(question_inputs.input_ids.shape[0]):
    selection.append(
        torch.flatten(mask_arr[i].nonzero()).tolist()
    )

# Then apply these indices to each respective row in input_ids, assigning each of 
# the values at these indices as 103.
for i in range(question_inputs.input_ids.shape[0]):
    question_inputs.input_ids[i, selection[i]] = 103

# create dataset
class QuestionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

question_dataset = QuestionDataset(question_inputs)
loader = torch.utils.data.DataLoader(question_dataset, batch_size=16, shuffle=True)
# use hugging face trainer 
args = TrainingArguments(
    # output_dir='/common/scratch/CS425/CS425G7',
    per_device_train_batch_size=16,
    num_train_epochs=EPOCHS
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=question_dataset
)

trainer.train()
print("finished training")
save_path = f"{MODEL_PATH}/{TEST_NAME}.pth"
torch.save(model.state_dict(), save_path)

