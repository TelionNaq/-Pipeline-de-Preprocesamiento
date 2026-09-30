# Informe Técnico: Pipeline de Preprocesamiento y Tokenización en NLP

## 1. Definición del Problema Real
**Detección y Clasificación de Reclamos Críticos en Atención al Cliente (E-commerce)**

En la industria del e-commerce, el volumen de interacciones con soporte al cliente (tickets, correos, chats) es inmenso. El problema radica en que **no todos los mensajes tienen la misma prioridad**. Un cliente consultando por horarios de atención no tiene la misma urgencia que un cliente enfurecido porque su pedido no llegó y se le cobró doble. 

Automatizar la clasificación de estos mensajes como **Urgentes**, **Neutros** o **Positivos** permite enrutar inmediatamente los casos críticos a operadores humanos especializados, mitigando el riesgo reputacional y la insatisfacción.

**El Desafío del Dato:** Los clientes escriben de forma ruidosa: errores tipográficos ("kiero"), mayúsculas sostenidas ("URGENTE"), excesivos signos de puntuación ("!!!!"), modismos y emojis. Para entrenar un modelo de Machine Learning, estos datos textuales no estructurados deben convertirse en representaciones tensoriales densas numéricas.

---

## 2. Arquitectura de la Solución (Pipeline)

El desarrollo abarca las fases previas al entrenamiento y clasificación:

1. **Adquisición del Dataset**: Carga de textos crudos y su etiqueta objetivo.
2. **Preprocesamiento y Limpieza**: Normalización a minúsculas, limpieza de URLs y remoción de caracteres espurios manteniendo la puntuación que indica sentimiento (como `¡` o `?`).
3. **Tokenización (Codificación Numérica)**: Transformación del string a secuencias de enteros (IDs), estandarizando longitudes (Truncamiento + Padding).

---

## 3. Justificación de los Enfoques de Tokenización

Se implementaron dos estrategias comparativas:

### A. Enfoque Clásico (Tokenización Léxica - Nivel de Palabra)
Este paradigma (frecuente en modelos como TF-IDF o Redes Neuronales Recurrentes clásicas) divide el texto mediante espacios y signos de puntuación. 
* **Ventaja:** Sencillo de implementar, rápido, vocabulario fácil de interpretar.
* **Desventaja:** Limitado al problema de *Out-Of-Vocabulary* (OOV). Si una palabra tiene un error tipográfico (ej. "urgneteee"), el tokenizador la mapeará al token `<UNK>` (desconocido), perdiendo información semántica vital. Además, el tamaño del vocabulario puede crecer desmesuradamente.

### B. Enfoque Moderno (Subwords con Hugging Face)
Utilizamos un modelo BPE / WordPiece (como **BETO - BERT en Español**). Este enfoque divide el texto en "subpalabras". Si encuentra una palabra desconocida, la fragmenta (ej. `urg`, `##ente`).
* **Ventaja:** Resuelve el problema OOV de manera elegante sin inflar el diccionario. Mantiene la estructura del texto intacto, codificando prefijos, raíces y sufijos.
* **Outputs:** 
    - `input_ids`: Tensor con los índices enteros de los subwords.
    - `attention_mask`: Tensor de ceros y unos para indicar al modelo qué tokens son reales y cuáles son rellenos (`[PAD]`), evitando que el modelo preste atención al padding.

---

## 4. Estructura de Salida
Los tensores generados al final del pipeline en el enfoque moderno tienen la estructura `(batch_size, sequence_length)` (en nuestra prueba `(8, 32)`). Esta estructura matricial está matemáticamente lista para ser inyectada a una capa de Embedding (Ej. `nn.Embedding` en PyTorch) de una red profunda clasificadora.

## 5. Instrucciones de Ejecución

Para replicar este ambiente e iniciar el pipeline, es necesario tener instalado Python 3.10+ y ejecutar en terminal:

```bash
# 1. Instalar las dependencias necesarias:
pip install -r requirements.txt

# 2. Ejecutar el orquestador principal:
python main.py
```
