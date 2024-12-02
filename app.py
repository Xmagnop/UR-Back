import os
from flask import Flask, request, send_file, jsonify, url_for
from ultralytics import YOLO
from PIL import Image
from werkzeug.utils import secure_filename

# Configuração do Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['OUTPUT_FOLDER'] = './outputs'
app.config['MODEL_FOLDER'] = './models'
AQUATIC_MODEL_PATH = os.path.join(
    app.config['MODEL_FOLDER'], 'aquatic_animals_model.pt')
PRETRAINED_MODEL = 'yolov5s.pt'

# Criar pastas necessárias
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Verificar se o modelo está disponível
if not os.path.exists(AQUATIC_MODEL_PATH):
  raise FileNotFoundError(
      f"Modelo ajustado não encontrado em {AQUATIC_MODEL_PATH}. Execute o train.py primeiro.")

# Carregar os modelos
objects_model = YOLO(PRETRAINED_MODEL)  # Modelo pré-treinado para objetos
# Modelo ajustado para animais aquáticos
animals_model = YOLO(AQUATIC_MODEL_PATH)

# Função para processar imagens e desenhar caixas delimitadoras


def process_image_with_boxes(image_path, models):
  combined_results = []

  # Processar imagem com cada modelo e combinar resultados
  for model in models:
    results = model(image_path)
    combined_results.append(results)

  # Renderizar todas as predições na imagem
  output_path = os.path.join(
      app.config['OUTPUT_FOLDER'], os.path.basename(image_path))
  image = Image.open(image_path).convert("RGB")

  for results in combined_results:
    results[0].plot()  # Desenha as predições diretamente na imagem

  # Salva a imagem processada
  image.save(output_path)
  return output_path


@app.route('/predict', methods=['POST'])
def predict():
  if 'image' not in request.files:
    return jsonify({'error': 'Nenhuma imagem enviada'}), 400

  # Salvar a imagem enviada
  file = request.files['image']

  # Garantir que o arquivo tenha um nome válido
  filename = secure_filename(
      file.filename) if file.filename else 'default_image.jpg'

  file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
  file.save(file_path)

  # Processar a imagem com os dois modelos
  output_path = process_image_with_boxes(
      file_path, [objects_model, animals_model])

  # Retornar a URL da imagem processada
  output_url = url_for('get_processed_image',
                       filename=os.path.basename(output_path), _external=True)
  return jsonify({'processed_image_url': output_url})


@app.route('/outputs/<filename>')
def get_processed_image(filename):
  # Endpoint para servir a imagem processada
  file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
  if not os.path.exists(file_path):
    return jsonify({'error': 'Arquivo não encontrado'}), 404
  return send_file(file_path, mimetype='image/jpeg')


if __name__ == '__main__':
  app.run(debug=True)
