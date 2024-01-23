import PyPDF2

def leer_pdf(nombre_archivo):
    # Abre el archivo PDF en modo de lectura binaria
    with open(nombre_archivo, 'rb') as archivo:
        # Crea un objeto lector de PDF
        lector_pdf = PyPDF2.PdfReader(archivo)
        
        # Inicializa una cadena para almacenar el contenido del PDF
        contenido_pdf = ""
        
        # Itera sobre todas las páginas del PDF
        for pagina_numero in range(len(lector_pdf.pages)):
            # Obtiene la página actual
            pagina = lector_pdf.pages[pagina_numero]
            
            # Extrae el texto de la página
            texto_pagina = pagina.extract_text()
            
            # Concatena el texto al contenido total
            contenido_pdf += texto_pagina
        
        return contenido_pdf
    
def split_into_batches(text, words_per_batch=70):
    # Split the string into words
    words = text.split()

    # Initialize the list of batches
    batches = []

    # Iterate over the words and batch them
    for i in range(0, len(words), words_per_batch):
        batch = words[i:i + words_per_batch]
        batches.append(' '.join(batch))

    return batches
# Reemplaza 'ejemplo.pdf' con el nombre de tu archivo PDF
pdf_path = '/home/lucasbiagetti/Documentos/gitPr/vectorizer/sample_data/communist_manifest.pdf'

# Llama a la función para leer el PDF y asigna el contenido a una variable
contenido_del_pdf = leer_pdf(pdf_path)



# Imprime el contenido del PDF
batches = split_into_batches( contenido_del_pdf)
for batch in batches:
    print (batch)
    print ("---")

