import os
import logging
import nltk
import mlflow
import mlflow.pyfunc
from datasets import load_dataset
from transformers import AutoTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
nltk.download("punkt")

def fine_tune_summarization_model():
    logger.info("Loading CNN/DailyMail dataset...")
    dataset = load_dataset("cnn_dailymail", "3.0.0")

    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    
    def preprocess_function(examples):
        inputs = [doc for doc in examples["article"]]
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["highlights"], max_length=128, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    
    train_dataset = tokenized_datasets["train"].select(range(2000))
    eval_dataset = tokenized_datasets["validation"].select(range(200))

    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

    training_args = TrainingArguments(
        output_dir="./training_output",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    logger.info("Starting summarization model fine-tuning...")
    trainer.train()

    return trainer, tokenizer

if __name__ == "__main__":
    with mlflow.start_run(run_name="summarization_training"):
        trainer, tokenizer = fine_tune_summarization_model()
        metrics = trainer.evaluate()
        mlflow.log_metrics(metrics)
        
        mlflow.pyfunc.log_model(
            python_model=trainer.model,
            artifact_path="summarization-model-artifact",
            registered_model_name="bart-summarization-model",
        )
        logger.info("Model training and logging completed.")    