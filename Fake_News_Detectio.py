import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset
from torch import nn
from torch.utils.data import DataLoader

# Load the datasets
dev_path = 'Fake_dev.csv'
train_path = 'Fake_train.csv'
test_path = 'Fake_test_without_labels.csv'

dev_data = pd.read_csv(dev_path)
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Preprocess the datasets
def preprocess_data(data):
    data = data.copy()
    if 'label' in data.columns:
        data['label'] = data['label'].map({'Fake': 0, 'original': 1})
    return data

train_data = preprocess_data(train_data)
dev_data = preprocess_data(dev_data)
test_data = preprocess_data(test_data)

# Convert to HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_data)
dev_dataset = Dataset.from_pandas(dev_data)
test_dataset = Dataset.from_pandas(test_data)

# Model names
model_names = ["distilbert-base-uncased", "xlm-roberta-base", "bert-base-multilingual-cased"]

# Define LSTM Model
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        lstm_out, (hidden, _) = self.lstm(embedded)
        output = self.fc(hidden[-1])
        return output

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()
    return cm

# Model comparison table
def create_comparison_table(models_metrics, model_names):
    comparison_df = pd.DataFrame(models_metrics, index=model_names, 
                                 columns=['Precision', 'Recall', 'F1-Score', 'Accuracy'])
    return comparison_df

# Error analysis helper
def error_analysis(y_true, y_pred, labels, model_name):
    cm = confusion_matrix(y_true, y_pred)
    errors = []
    for i in range(len(labels)):
        fn = cm[i, :].sum() - cm[i, i]  # False Negatives
        fp = cm[:, i].sum() - cm[i, i]  # False Positives
        errors.append({
            'Class': labels[i],
            'False Positives': fp,
            'False Negatives': fn
        })
    error_df = pd.DataFrame(errors)
    print(f"\nError Analysis for {model_name}:")
    print(error_df)
    return error_df

# Training and evaluation for Transformers models
def train_and_evaluate_transformer(model_name, train_dataset, eval_dataset, num_labels=2):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    
    training_args = TrainingArguments(
        output_dir=f"./Final Fake News/{model_name}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",  
        greater_is_better=True,
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
        accuracy = (preds == labels).mean()
        return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    predictions = trainer.predict(eval_dataset)
    return predictions.predictions.argmax(-1), eval_dataset['label'], trainer

# Training and evaluation for LSTM
def train_and_evaluate_lstm(train_dataset, eval_dataset, embedding_dim=128, hidden_dim=256, output_dim=2):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    
    train_dataset.set_format('torch', columns=['input_ids', 'label'])
    eval_dataset.set_format('torch', columns=['input_ids', 'label'])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=16)
    
    model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(5):
        model.train()
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids)
            preds.extend(outputs.argmax(dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    return np.array(preds), np.array(true_labels), model

# Main execution
labels = ['Fake', 'original']
models_metrics = []
confusion_matrices = []

# Train and evaluate LSTM
print("\nTraining LSTM")
lstm_preds, lstm_true, _ = train_and_evaluate_lstm(train_dataset, dev_dataset)
cm_lstm = plot_confusion_matrix(lstm_true, lstm_preds, labels, "Confusion Matrix - LSTM")
precision, recall, f1, _ = precision_recall_fscore_support(lstm_true, lstm_preds, average='macro')
accuracy = (lstm_preds == lstm_true).mean()
models_metrics.append([precision, recall, f1, accuracy])
error_analysis(lstm_true, lstm_preds, labels, "LSTM")
print("\nClassification Report for LSTM:")
print(classification_report(lstm_true, lstm_preds, target_names=labels))

# Train and evaluate Transformer models
for model_name in model_names:
    print(f"\nTraining {model_name}")
    preds, true, _ = train_and_evaluate_transformer(model_name, train_dataset, dev_dataset)
    cm = plot_confusion_matrix(true, preds, labels, f"Confusion Matrix - {model_name}")
    confusion_matrices.append(cm)
    precision, recall, f1, _ = precision_recall_fscore_support(true, preds, average='macro')
    accuracy = (preds == true).mean()
    models_metrics.append([precision, recall, f1, accuracy])
    error_analysis(true, preds, labels, model_name)
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(true, preds, target_names=labels))

# Comparison Table
comparison_table = create_comparison_table(models_metrics, ['LSTM'] + model_names)
print("\nModel Comparison Table:")
print(comparison_table)

# Plot comparison graph 
comparison_table.plot(kind='line', figsize=(10, 6), marker='o')
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.legend(loc='best')
plt.grid(True)
plt.show()