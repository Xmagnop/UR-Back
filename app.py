import os
from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
import torch
from PIL import Image

# Configuração do Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['OUTPUT_FOLDER'] = './outputs'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Carregar os modelos
objects_model = torch.hub.load(
    'ultralytics/yolov5', 'yolov5s')  # YOLOv5 para objetos
animals_model = torch.hub.load(
    'ultralytics/yolov5', 'custom', path='./models/aquatic_animals_model.pth')


# Função para processar imagens e desenhar caixas delimitadoras
def process_image_with_boxes(image_path, models):
  image = Image.open(image_path).convert("RGB")
  combined_results = []

  # Processar imagem em cada modelo e combinar resultados
  for model in models:
    results = model(image)
    combined_results.append(results)

  # Renderizar as predições sobre a imagem
  combined_image = image.copy()
  for result in combined_results:
    result.save(save_dir=app.config['OUTPUT_FOLDER'], exist_ok=True)

  # Retorna o caminho da imagem processada
  output_path = os.path.join(
      app.config['OUTPUT_FOLDER'], os.path.basename(image_path))
  return output_path


@app.route('/predict', methods=['POST'])
def predict():
  if 'image' not in request.files:
    return jsonify({'error': 'Nenhuma imagem enviada'}), 400

  # Salvar a imagem enviada
  file = request.files['image']
  filename = secure_filename(file.filename)
  file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
  file.save(file_path)

  # Processar a imagem com ambos os modelos
  output_path = process_image_with_boxes(
      file_path, [objects_model, animals_model])

  # Retornar a imagem processada
  return send_file(output_path, mimetype='image/jpeg')


if __name__ == '__main__':
  app.run(debug=True)
