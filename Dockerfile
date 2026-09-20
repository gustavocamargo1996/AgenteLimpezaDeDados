FROM python:3.12-slim

WORKDIR /app

# Camada de dependencias separada: muda pouco, cacheia bem.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# So' os dois arquivos que o Embedder abre.
COPY all-MiniLM-L6-v2/onnx/model_O4.onnx  /modelo/onnx/model_O4.onnx
COPY all-MiniLM-L6-v2/tokenizer.json      /modelo/tokenizer.json
ENV MODELO_EMBEDDING=/modelo

COPY limpeza/ ./limpeza/
COPY main.py avaliar_limpador.py escolher_modelo.py ./

# Codigo escrito por LLM roda aqui dentro: uid fixo nao-root, nunca 0.
# /dados e' o ponto de montagem do volume; criado e cedido antes do USER.
RUN useradd --uid 10001 --create-home limpeza \
    && mkdir -p /dados \
    && chown -R limpeza:limpeza /dados /app
USER limpeza

ENTRYPOINT ["python", "main.py"]
