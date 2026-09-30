# NLP Tokenization Pipeline: Customer Service Complaints

Este proyecto es una prueba de concepto para automatizar una tarea súper común en e-commerce: **procesar y clasificar mensajes de atención al cliente**. 

El alcance de este repositorio llega hasta la etapa de **preprocesamiento y tokenización**, dejando los datos de texto estructurados en tensores numéricos, totalmente listos para ser consumidos por un modelo de Machine Learning (como una red neuronal profunda o un modelo de clasificación).

## El Problema

En atención al cliente, no todos los mensajes tienen la misma urgencia. Alguien preguntando "¿a qué hora abren?" no es tan crítico como alguien diciendo "¡¡urgente!! mi paquete no llegó y me cobraron 2 veces". 
El desafío principal es que la gente escribe como quiere (errores tipográficos, exceso de signos de puntuación, mayúsculas, enlaces, etc). Este pipeline se encarga de recibir esa data cruda y ruidosa, limpiarla y transformarla en matemáticas para que un modelo la pueda procesar.

## Qué hace el código

Para el desarrollo, armé el pipeline implementando y comparando dos enfoques distintos:

1. **Enfoque Clásico (Word-Level):** Usando limpieza con regex y armado manual de vocabulario con padding. Funciona rápido, pero sufre cuando hay errores de ortografía (palabras desconocidas).
2. **Enfoque Moderno (Subwords con Hugging Face):** Utilizando `BETO` (un modelo BERT preentrenado en español). Este enfoque usa tokens a nivel de "sub-palabras" (WordPiece), lo que resuelve maravillosamente el problema del vocabulario sin perder el contexto de la frase original.

## Estructura del Repo

* `src/preprocessing.py`: Módulo que incluye funciones de normalización de texto y el tokenizador clásico.
* `src/hf_tokenizer.py`: Implementación rápida del pipeline usando Transformers.
* `main.py`: Script principal para levantar un dataset de muestra (en memoria) y pasar los textos por ambos pipelines para ver la transformación.
* `INFORME_TECNICO.md`: Documento con la justificación teórica y técnica del enfoque (versión formal del entregable).

## Cómo probarlo en local

Necesitas Python 3.10 o superior.

1. Instala los requerimientos:
   ```bash
   pip install -r requirements.txt
   ```

2. Ejecuta el archivo principal:
   ```bash
   python main.py
   ```

Al final de la ejecución, vas a ver por consola cómo se generan los `input_ids` y la `attention_mask` con la forma matricial exacta (tensores) necesaria para el entrenamiento.
