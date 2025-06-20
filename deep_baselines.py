from utilities import pipeline
import pandas as pd
import argparse
import torch
import copy

def test(evaluation_type, task, architecture_mode, PRE_TRAINED_MODEL_NAME_TEXT, PRE_TRAINED_MODEL_NAME_VISION, MAX_LENGTH_TEXT, encoding_dimension, save_labels):
    pipe = pipeline.pipeline(evaluation_type, task, architecture_mode, PRE_TRAINED_MODEL_NAME_TEXT, PRE_TRAINED_MODEL_NAME_VISION, MAX_LENGTH_TEXT, encoding_dimension, save_labels)
    
    path_text, path_img, labels, target_names, N_CLASSES, EPOCHS, BATCH_SIZE, LEARNING_RATE = pipe.get_hyperparameters()
    
    x_train_text, x_train_img, y_train, x_test_text, x_test_img, labels_train = pipe.get_full_dataset(path_text)
    
    if architecture_mode == "EF" or architecture_mode == "ViT":
        img_tensors_train, img_tensors_test = pipe.get_image_tensors(path_img, x_train_img, x_test_img)
    else:
        img_tensors_train = img_tensors_test = None
    
    train_dataloader, test_dataloader = pipe.tokenize_full_dataset(x_train_text, img_tensors_train, y_train, x_test_text, img_tensors_test, labels_train, BATCH_SIZE)
    
    model, n_params, device = pipe.load_model(N_CLASSES)
    
    print(f"Total parameters: {n_params}")
    
    t_handler = pipe.train_model(model, device, train_dataloader, EPOCHS, LEARNING_RATE)
    
    model_prediction_test = pipe.evaluate(t_handler, model, test_dataloader, N_CLASSES)
    
    return model_prediction_test

def main(args):
    task                            = int(args.task) # 1, 2
    evaluation_type                 = args.evaluation_type
    architecture_mode               = args.architecture_mode # BETO, ViT, EF
    
    PRE_TRAINED_MODEL_NAME_TEXT     = "dccuchile/bert-base-spanish-wwm-uncased"
    PRE_TRAINED_MODEL_NAME_VISION   = "google/vit-base-patch16-224-in21k"
    
    # Define some network parameters
    encoding_dimension              = 768 # Define the encoding dimension (according to BERT) 768
    MAX_LENGTH_TEXT                 = 40 # Define the maximum length of the sequence
    
    save_labels                   = True
    
    iterations                    = 1
    
    for i in range(iterations):
        model_prediction_test = test(evaluation_type, task, architecture_mode, PRE_TRAINED_MODEL_NAME_TEXT, PRE_TRAINED_MODEL_NAME_VISION, MAX_LENGTH_TEXT, encoding_dimension, save_labels)
        if save_labels:
            df = pd.DataFrame(model_prediction_test)
            df.to_csv("results/" + architecture_mode + "_baseline_test_results" + "_subtask_" + str(task) + ".csv", index=False, header=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation_type", type=str, required=True)
    parser.add_argument("--architecture_mode", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    args = parser.parse_args()
    main(args)