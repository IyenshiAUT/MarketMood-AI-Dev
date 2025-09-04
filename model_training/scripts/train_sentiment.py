import os
import logging
import numpy as np
import mlflow
import mlflow.pyfunc
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fine_tune_sentiment_model():
    logger.info("Loading financial phrasebank dataset...")
    dataset = load_dataset("financial_phrasebank", "sentences_allagree")
    split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
    
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    
    def tokenize_function(examples):
        return tokenizer(examples['sentence'], truncation=True, padding=True)
    
    tokenized_dataset = split_dataset.map(tokenize_function, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", num_labels=3)
    
    training_args = TrainingArguments(
        output_dir="./training_output",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=lambda p: {"eval_f1": f1_score(p.label_ids, np.argmax(p.predictions, axis=1), average="weighted")},
        tokenizer=tokenizer
    )
    
    logger.info("Starting sentiment model fine-tuning...")
    trainer.train()
    
    return trainer

if __name__ == "__main__":
    with mlflow.start_run(run_name="sentiment_training"):
        trainer = fine_tune_sentiment_model()
        metrics = trainer.evaluate()
        mlflow.log_metrics(metrics)
        
        mlflow.pyfunc.log_model(
            python_model=trainer.model,
            artifact_path="sentiment-model-artifact",
            registered_model_name="finbert-sentiment-model",
        )
        logger.info("Model training and logging completed.")