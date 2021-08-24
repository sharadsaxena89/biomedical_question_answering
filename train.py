import json
from transformers import DistilBertTokenizerFast
import random
import torch

list_of_linearized_tables = []

# Linearized table contents are stored in the file final_cleaned_linearized_table_contents.json
# They are stored separately from question-answer dataset file, since file size was getting too big, if we store context
# along with question, answers
# We have chosen to combine context from linearzied table at time of training
# The linearized tables will be stored in the variable list_of_linearized_tables
f2 = open('/content/drive/My Drive/question_answer_dataset/final_cleaned_linearized_table_contents.json', 'r', errors="ignore")
json_object = json.load(f2)

token_count = 0
count = 0
max = 0
for table in json_object: 
    list_of_linearized_tables.append(table['contents'])
    
f2.close()

# final_question_answer_dataset.json is the file which contains questions and answers
# This is combined with context that we have extracted earlier
with open('/content/drive/My Drive/question_answer_dataset/final_question_answer_dataset.json', "r", errors="ignore") as reader:
  input_data = json.load(reader)

print('Unpacking Question Answer Dataset...')

print('Tables:')

list_of_entries = []

complete_train_contexts = []
complete_train_questions = []
complete_train_answers = []

question_id = 0

for entry in input_data:
    
    title = entry["table_name"]
    answer = {"answer_start": int(entry["answer_start"]), "answer_end": int(entry["answer_end"]),"text": entry["answer"]} 

    current_entry = {"question": entry["question"], "answer": answer, "title": title}
    list_of_entries.append(current_entry) 

# Shuffle the data
random.shuffle(list_of_entries)

for entry in list_of_entries:
  title = entry['title']
  complete_train_contexts.append(list_of_linearized_tables[int(title)-1])
  complete_train_questions.append(entry['question'])
  complete_train_answers.append(entry['answer'])

print('DONE!')

# Next we need to convert our character start/end positions to token start/end positions
# DEFINING function to add start/end positions
# Datset too large, so this function will be called multiple times
def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))

        # Log, can be removed
        # print(encodings.char_to_token(i, answers[i]['answer_start']))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length

    # print(len(start_positions))
    # print(len(end_positions))
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})
    

# Overriding some methods of the encodings, so that tensors containing input_ids, attention masks, start and end positon
# can be retrieved
class QADataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)
      
# Epoch loop is outside
# Import the tokenizer
from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Import the model
from transformers import DistilBertForQuestionAnswering
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

from torch.utils.data import DataLoader
from transformers import AdamW

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)
model.train()

for epoch in range(3):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, 3)) 

    # Construct the train dataset
    for start_index in range(0,798001,2000):
      end_index = start_index + 2000

      train_questions =  complete_train_questions[start_index:end_index]
      train_contexts = complete_train_contexts[start_index:end_index]
      train_answers = complete_train_answers[start_index:end_index]

      print("Tokenizing entries: {:} to {:}".format(start_index,end_index))
     
      train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
      add_token_positions(train_encodings, train_answers)

      train_dataset = QADataset(train_encodings)

      train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
      optim = AdamW(model.parameters(), lr=5e-5) 
     # Train dataset has been constructed
      for batch in train_loader:
        # Log
        total_train_loss = 0

        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs[0]

        # Log
        log_loss = outputs.loss
        total_train_loss += log_loss.item() 

        loss.backward()
        optim.step()
       # Log
      avg_train_loss = total_train_loss / len(train_loader)
      print("")
      print("  Average training loss: {0:.2f}".format(avg_train_loss))

model.eval()
    
