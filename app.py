import os
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import yolov5
from PIL import Image
import numpy as np
import cv2
from werkzeug.utils import secure_filename
import tempfile

# Configuração do Flask
app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = './uploads'
app.config['OUTPUT_FOLDER'] = './outputs'
app.config['MODEL_FOLDER'] = './models'
AQUATIC_MODEL_PATH = 'best.pt'
PRETRAINED_MODEL = 'yolov5s.pt'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Verificar se o modelo está disponível
if not os.path.exists(AQUATIC_MODEL_PATH):
    raise FileNotFoundError(
        f"Modelo ajustado não encontrado em {AQUATIC_MODEL_PATH}. Execute o train.py primeiro.")

# Carregar modelos
objects_model = yolov5.load(PRETRAINED_MODEL)  # Modelo pré-treinado para objetos (Não utilizado)
animals_model = yolov5.load(AQUATIC_MODEL_PATH)  # Modelo ajustado para animais aquáticos

# Processar imagens e gerar detecções
def process_image_with_boxes(image_path, models):
    combined_results = []

    # Processar imagem com cada modelo e combinar resultados
    for model in models:
        results = model(image_path)  # Realizar predição
        combined_results.append(results)

    # Abrir a imagem usando PIL e converter para formato NumPy
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Para cada conjunto de resultados, desenhamos os quadrados nas imagens
    for results in combined_results:
        # 'xyxy' contém as coordenadas dos quadrados (x1, y1, x2, y2)
        boxes = results.xyxy[0].cpu().numpy()  # Pega os quadrados das predições
        confidences = results.pred[0][:, 4].cpu().numpy()  # Accuracy
        labels = results.names  # Labels (nomes)
        class_ids = results.pred[0][:, -1].cpu().numpy()  # IDs

        for i in range(len(boxes)):
            # Garantir que estamos acessando corretamente as 4 coordenadas do quadrado
            # O formato dos quadrados é [x1, y1, x2, y2, confidence]
            box = boxes[i][:4]  # Pegamos as 4 primeiras coordenadas: [x1, y1, x2, y2]
            x1, y1, x2, y2 = box  # Extraímos as 4 coordenadas da caixa

            class_id = int(class_ids[i])
            confidence = confidences[i]

            # Desenha o quadrado
            color = (0, 255, 0)  # Cor verde para os quadrados
            thickness = 2  # Espessura do quadrado
            cv2.rectangle(image_np, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

            # Escreve label
            text = f"{labels[class_id]} {confidence:.2f}"
            cv2.putText(image_np, text, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Salvar a imagem com quadrados
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'processed_image.jpg')
    output_image = Image.fromarray(image_np)
    output_image.save(output_path)
    return output_path

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Nenhuma imagem enviada'}), 400

    # Salvar a imagem enviada
    file = request.files['image']

    # Garantir que o arquivo tenha um nome válido
    filename = secure_filename(file.filename) if file.filename else 'default_image.jpg'

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Processar a imagem
    output_path = process_image_with_boxes(file_path, [ animals_model])

    # Retornar a URL da imagem processada
    return jsonify({'message': 'Imagem processada com sucesso. Acesse o arquivo em /outputs'})


@app.route('/outputs', methods=['GET'])
def get_image_nf():
    # O arquivo processado será sempre 'processed_image.jpg'
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], 'processed_image.jpg')

    if not os.path.exists(file_path):
        return jsonify({'error': 'Arquivo não encontrado'}), 404

    # Enviar a imagem processada como arquivo
    return send_file(file_path, 
                     as_attachment=True,
                     download_name='processed_image.jpg',
                     mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
