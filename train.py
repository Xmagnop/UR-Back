import os
import subprocess
import kagglehub
import yaml
from pathlib import Path

# Baixar o dataset com o kagglehub
print("Baixando o dataset...")
dataset_path = kagglehub.dataset_download("slavkoprytula/aquarium-data-cots")
print(f"Dataset baixado em: {dataset_path}")

# Caminhos para os arquivos
PROJECT_DIR = Path(__file__).resolve().parent
# Usando o data.yaml do dataset
DATA_YAML_PATH = Path(f'{dataset_path}/aquarium_pretrain/data.yaml')
dataset_path = f'{dataset_path}/aquarium_pretrain'

# Função para corrigir os caminhos no arquivo data.yaml


def fix_data_yaml(data_yaml_path, dataset_path):
  with open(data_yaml_path, 'r') as file:
    data = yaml.safe_load(file)

  # Ajustar os caminhos das imagens e labels
  data['train'] = os.path.join(dataset_path, 'train', 'images')
  data['val'] = os.path.join(dataset_path, 'valid', 'images')
  data['test'] = os.path.join(dataset_path, 'test', 'images')

  # Salvar as modificações no data.yaml
  with open(data_yaml_path, 'w') as file:
    yaml.dump(data, file)

# Função para treinar o modelo


def train_model():
  try:
    print("Iniciando treinamento do modelo YOLOv5 com o dataset...")

    # Verificar se o data.yaml existe
    if not DATA_YAML_PATH.exists():
      print(f"Erro: {DATA_YAML_PATH} não encontrado.")
      return

    # Corrigir os caminhos dentro do data.yaml
    fix_data_yaml(DATA_YAML_PATH, dataset_path)

    # Chamar o script de treinamento do YOLOv5
    print("Executando o treinamento com o script oficial do YOLOv5...")

    # Defina os parâmetros para o treinamento
    epochs = 50
    batch_size = 16
    imgsz = 640

    # Chama o script de treinamento oficial do YOLOv5 via subprocess
    subprocess.run([
        'python', 'yolov5/train.py',  # Caminho do script de treinamento
        # Caminho do arquivo data.yaml corrigido
        '--data', str(DATA_YAML_PATH),
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--img-size', str(imgsz),
        '--weights', 'yolov5s.pt',  # Usar o modelo YOLOv5s pré-treinado
        '--cache'  # Cache as imagens para acelerar o treinamento
    ])

    print("Treinamento concluído!")
  except Exception as e:
    print(f"Erro durante o treinamento: {e}")


if __name__ == '__main__':
  train_model()
