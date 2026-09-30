import pandas as pd
from src.preprocessing import TextPreprocessor, ClassicalTokenizer
from src.hf_tokenizer import ModernTokenizer

def main():
    print("="*80)
    print("PIPELINE DE NLP: DETECCIÓN DE RECLAMOS CRÍTICOS EN E-COMMERCE")
    print("="*80)
    
    # ---------------------------------------------------------
    # 1. Dataset de Muestra (Problema Real de E-commerce)
    # ---------------------------------------------------------
    print("\n[1] Creando dataset representativo de muestra...")
    data = [
        "¡URGENTE! Mi pedido #98421 no ha llegado y el cobro ya se aplicó dos veces. ¿Me pueden dar una solución ya??? https://tienda.com/p/98421",
        "Hola, quería saber el horario de atención de la sucursal centro. Gracias.",
        "Excelente servicio, el producto llegó antes de lo esperado. ¡Muchas felicidades!",
        "Pésimo!!!! El paquete llego roto y nadie contesta los correos. Exijo un reembolso.",
        "Necesito cambiar la dirección de envío del pedido 12345, ¿aún estoy a tiempo?",
        "donde sta mi pedido??? es una estafa kiero mi plata ya mismo ahhhh!!",
        "Buenas tardes. Realicé una compra hace 5 días pero el estado sigue en 'procesando'.",
        "El producto no es lo que esperaba, quiero iniciar una devolución."
    ]
    labels = ["Urgente", "Neutro", "Positivo", "Urgente", "Neutro", "Urgente", "Neutro", "Neutro"]
    
    df = pd.DataFrame({"texto_original": data, "etiqueta": labels})
    print("\nDataset cargado (3 primeros registros):")
    for idx, row in df.head(3).iterrows():
        print(f" - [{row['etiqueta']}] {row['texto_original']}")
    
    # ---------------------------------------------------------
    # 2. Preprocesamiento y Limpieza
    # ---------------------------------------------------------
    print("\n[2] Aplicando Preprocesamiento y Limpieza Textual...")
    preprocessor = TextPreprocessor()
    df["texto_limpio"] = df["texto_original"].apply(preprocessor.clean_text)
    
    texto_ejemplo_original = df['texto_original'].iloc[0]
    texto_ejemplo_limpio = df['texto_limpio'].iloc[0]
    
    print("\nTransformación del Ejemplo 1:")
    print(f"Crudo : {texto_ejemplo_original}")
    print(f"Limpio: {texto_ejemplo_limpio}")
    
    texto_sin_stopwords = preprocessor.remove_stopwords(texto_ejemplo_limpio)
    print(f"Sin SW: {texto_sin_stopwords} (Demostración de Stopwords)")
    
    # ---------------------------------------------------------
    # 3. Tokenización Clásica (Nivel de palabra)
    # ---------------------------------------------------------
    print("\n[3] Tokenización Clásica (Word-Level + Padding + Vocabulario)...")
    classic_tok = ClassicalTokenizer(max_vocab_size=50)
    classic_tok.build_vocab(df["texto_limpio"].tolist())
    
    df["classic_encoded"] = df["texto_limpio"].apply(lambda x: classic_tok.tokenize_and_encode(x, max_length=20))
    
    print("Vector de IDs Clásico (Max_len=20):")
    print(df["classic_encoded"].iloc[0])
    
    # ---------------------------------------------------------
    # 4. Tokenización Moderna (Subwords con Transformers)
    # ---------------------------------------------------------
    print("\n[4] Tokenización Moderna (Subword/WordPiece con Hugging Face)...")
    # Utilizamos BETO para tener soporte especializado en español
    modern_tok = ModernTokenizer("dccuchile/bert-base-spanish-wwm-cased")
    
    # Los modelos basados en Transformers son lo suficientemente robustos para procesar el texto original con mínimos cambios, 
    # pero utilizamos un max_length adaptado.
    modern_output = modern_tok.process_texts(df["texto_original"].tolist(), max_length=32)
    
    print("\n--- RESULTADOS DEL TOKENIZADOR DE TRANSFORMERS PARA EL EJEMPLO 1 ---")
    print("\nTexto de entrada:")
    print(texto_ejemplo_original)
    
    print("\nTokens extraídos (Nótese los fragmentos ##):")
    print(modern_output["tokens"][0])
    
    print("\nInput IDs (Vector Numérico):")
    print(modern_output["input_ids"][0].tolist())
    
    print("\nAttention Mask (Apunta a tokens reales vs Padding):")
    print(modern_output["attention_mask"][0].tolist())
    
    print("\nEstructura (Shape) del tensor final listo para el Modelo:")
    print(f"Tamaño del tensor input_ids: {modern_output['input_ids'].shape} -> (Batch_size, Sequence_length)")
    
    print("\n" + "="*80)
    print("¡PIPELINE COMPLETADO! LOS TENSORES ESTÁN LISTOS PARA EMBEDDING/CLASIFICACIÓN.")
    print("="*80)

if __name__ == "__main__":
    main()
