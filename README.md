# Aquatic Animals Detection API

Este projeto implementa uma API Flask para detectar objetos e animais aquáticos em imagens usando o modelo YOLO. A API permite o envio de uma imagem, que é processada com dois modelos YOLO (um pré-treinado e outro ajustado para detectar animais aquáticos), e retorna uma imagem com caixas delimitadoras desenhadas em torno dos objetos detectados.

## Estrutura do Projeto
```
.
├── app.py                      # Código principal da API Flask
├── uploads/                    # Pasta onde as imagens enviadas são armazenadas
├── outputs/                    # Pasta onde as imagens processadas são armazenadas
├── models/                     # Pasta onde os modelos YOLO são armazenados
│   └── aquatic_animals_model.pt # Modelo ajustado para animais aquáticos
└── requirements.txt            # Arquivo de dependências
```
## Requisitos

Este projeto requer Python 3.7 ou superior. As dependências podem ser instaladas com o arquivo `requirements.txt`:

```
pip install -r requirements.txt
```
## Dependências:

* Flask
* ultralytics (para o YOLO)
* Pillow (para manipulação de imagens)
* werkzeug (para manipulação de arquivos)

## Como usar

### 1. Preparação

Antes de executar a aplicação, você precisa garantir que o modelo ajustado para animais aquáticos esteja disponível. Caso contrário, o servidor não será iniciado.

* Coloque o modelo ajustado `aquatic_animals_model.pt` na pasta `models`.
* Caso não tenha o modelo ajustado, execute o script `train.py` para treiná-lo.

### 2. Rodando o Servidor

Execute o servidor Flask com o comando:

```
python app.py

```
O servidor estará disponível em `http://127.0.0.1:5000/`.

### 3. Enviando uma Imagem para Processamento

Para realizar a detecção de objetos e animais aquáticos, envie uma imagem usando o endpoint `/predict` com um método `POST` e o arquivo da imagem como `multipart/form-data`.

Exemplo de requisição:

```
curl -X POST -F "image=@path_to_image.jpg" http://127.0.0.1:5000/predict
```

### 4. Resposta da Requisição

A resposta será um JSON com a URL da imagem processada:

```
{
  "processed_image_url": "http://127.0.0.1:5000/outputs/processed_image.jpg"
}
```
### 5. Obtendo a Imagem Processada

Para visualizar a imagem processada, acesse a URL fornecida pela resposta da requisição.

Exemplo:

```
http://127.0.0.1:5000/outputs/processed_image.jpg
```

A imagem processada conterá caixas delimitadoras desenhadas ao redor dos objetos detectados.

## Explicação do Código

### Configuração Inicial

* O Flask é configurado com as pastas `uploads`, `outputs`, e `models` para armazenar imagens enviadas, imagens processadas e modelos, respectivamente.
* O modelo YOLO pré-treinado para objetos e o modelo ajustado para animais aquáticos são carregados.

### Endpoint `/predict`

1. Recebe uma imagem enviada via `POST`.
2. A imagem é salva na pasta `uploads`.
3. A imagem é processada pelos dois modelos YOLO, e as predições são desenhadas diretamente na imagem.
4. A imagem processada é salva na pasta `outputs`.
5. A URL da imagem processada é retornada.

### Endpoint `/outputs/<filename>`

Este endpoint serve as imagens processadas. Quando o usuário acessa a URL fornecida após o envio de uma imagem, a imagem processada é retornada.

## Possíveis Erros

* Erro de Arquivo Não Encontrado: Caso o modelo ajustado não esteja presente, o Flask levantará um erro `FileNotFoundError` e não iniciará o servidor.
* Erro de Arquivo de Imagem: Se nenhuma imagem for enviada na requisição, a resposta será:
```
{
  "error": "Nenhuma imagem enviada"
}
```
* Erro de Arquivo Processado Não Encontrado: Se a imagem processada não for encontrada na pasta `outputs`, a resposta será:
```
{
  "error": "Arquivo não encontrado"
}
```
