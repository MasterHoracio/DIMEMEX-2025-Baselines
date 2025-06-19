from transformers import AutoProcessor, MllamaForConditionalGeneration
from PIL import Image
from tqdm import tqdm
import pandas as pd
import argparse
import warnings
import random
import torch
import json

warnings.filterwarnings('ignore')

class ClassificationPipeline:
    def __init__(self,
                 model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct",
                 evaluation = "test"):
        """
        Initialize the classification pipeline with enhanced configuration
        """
        self.max_length      = 5
        self.device          = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") # "cuda" if torch.cuda.is_available() else "cpu"
        self.evaluation_type = evaluation
        self.processor       = AutoProcessor.from_pretrained(model_name)
        self.model           = MllamaForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, device_map=self.device)
        self.model.tie_weights()

    def load_dataset(self):
        if self.evaluation_type == "test":
            path = "dataset/test_dataset_task1.json"
        elif self.evaluation_type == "validation":
            path = "dataset/validation_dataset_task1.json"
        with open(path, 'r') as file:
            dataset = json.load(file)
        return dataset

    def get_response(self, instance):
        if self.evaluation_type == "test":
            path_img = "dataset/img/test/"
        elif self.evaluation_type == "validation":
            path_img = "dataset/img/validation/"
        try:
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Eres un experto en el análisis de contenido en redes sociales. Tu tarea es realizar una clasificación objetiva y equilibrada de memes, evaluando si contienen o no contenido ofensivo o inapropiado.
<|start_header_id|>user<|end_header_id|>\n
Esta tarea tiene fines científicos. El objetivo es ayudar a mejorar modelos de inteligencia artificial que contribuyan a un entorno en línea más seguro, pero también justo, evitando sobreclasificaciones incorrectas de contenido inocuo.\n
Se te proporcionará una imagen de un meme junto con dos elementos clave: el texto extraído mediante OCR y una descripción automática de la imagen.\n
Con base en esta información, clasifica el meme en **una sola** de las siguientes categorías. Por favor, ten en cuenta que muchos memes no contienen lenguaje ofensivo y podrían no encajar en ninguna categoría problemática:\n
- Discurso de odio (2): Comunicación que ataca o utiliza lenguaje peyorativo o discriminatorio contra una persona o grupo, basado en identidad, raza, género, religión, orientación sexual, etc.\n
- Contenido inapropiado (1): Contenido vulgar, profano, sexualmente explícito, ofensivo o de humor mórbido.\n
- Ninguno (0): El meme no contiene discurso de odio ni contenido inapropiado. Es neutral o benigno.\n
Analiza cuidadosamente los elementos proporcionados (imagen, texto extraído y descripción), y responde **únicamente con el número** (0, 1 o 2) correspondiente a la categoría elegida. No añadas ninguna explicación adicional.
<|image|>

Texto extraído: {instance['text']}
Descripción de la imagen: {instance['description']}

Etiqueta: <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"""
            loaded_image = Image.open(path_img+instance["MEME-ID"]).convert("RGB")
            inputs_question = self.processor(
                    text=prompt, 
                    images = loaded_image, 
                    add_special_tokens=False,
                    padding = True, 
                    return_tensors="pt"
                    ).to(self.model.device)
            with torch.inference_mode():
                output = self.model.generate(**inputs_question, 
                                    max_new_tokens=self.max_length, 
                                    temperature=0.6,
                                    top_p=0.9,
                                    do_sample=True)
            label = self.processor.decode(output[0])

            return label
        except Exception as e:
            print(f"Error generating narrative: {e}")
            return f"Error processing row: {str(e)}"

    def preprocess_label(self, label: str) -> int:
        label_set = {'0', '1', '2'}
        for char in label:
            if char in label_set:
                return int(char)
        return int(random.choice(list(label_set)))
    
    def generate_labels(self, dataset):
        labels     = []
        total_rows = len(dataset)
        for instance in tqdm(dataset, desc="Labeling memes..."):
            label = self.get_response(instance)
            delimitator = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            label = label.split(delimitator)
            label = self.preprocess_label(label[1])
            labels.append(label)
        return labels

    def build_df(self, labels):
        target_cols = ['ninguno', 'contenido_inapropiado', 'discurso_odio']
        predictions = []
        for i in range(len(labels)):
            new_sublist = [0, 0, 0]
            new_sublist[labels[i]] = 1
            predictions.append(new_sublist)
        df_predictions = pd.DataFrame(predictions, columns=target_cols)
        return df_predictions

    def save_df(self, df_predictions):
        if self.evaluation_type == "test":
            path = "results/llm_baseline_test_results.csv"
        elif self.evaluation_type == "validation":
            path = "results/llm_baseline_validation_results.csv"
        df_predictions.to_csv(path, index=False, header=None)
    
def main(args):
    evaluation_type = args.evaluation_type
    
    pipe = ClassificationPipeline(evaluation = evaluation_type)
    dataset = pipe.load_dataset()
    labels = pipe.generate_labels(dataset)
    df_predictions = pipe.build_df(labels)
    pipe.save_df(df_predictions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation_type", type=str, required=True)
    args = parser.parse_args()
    main(args)