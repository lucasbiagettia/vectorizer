import re

def split_sentences(text):
    # Regular expression to split the text into sentences
    regex = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')

    # Split the text into sentences
    sentences = regex.split(text)

    # Initialize a list to store processed sentences
    processed_sentences = []

    # Initialize a variable to keep track of word count in the current sentence
    current_word_count = 0

    # Iterate through each sentence and process according to the specified condition
    for sentence in sentences:
        words = sentence.split()
        num_words_sentence = len(words)

        # Check if the sentence has at least 30 words
        if num_words_sentence >= 20:
            # Add the current sentence to the list of processed sentences
            processed_sentences.append(sentence)
            # Reset the word count for the next sentence
            current_word_count = 0
        else:
            # Check if there are previous sentences to combine with the current one
            if processed_sentences:
                # Add the current sentence to the previous sentence
                processed_sentences[-1] += ' ' + sentence
                # Update the word count
                current_word_count += num_words_sentence
            else:
                # If there are no previous sentences, simply add the current sentence
                processed_sentences.append(sentence)
                # Update the word count
                current_word_count = num_words_sentence

    return processed_sentences

# Ejemplo de uso
texto_ejemplo = "Sé que me acusan de soberbia, y tal vez de misantropía, y tal vez de locura. Tales acusaciones (que yo castigaré a su debido tiempo) son irrisorias. Es verdad que no salgo de mi casa, pero también es verdad que sus puertas (cuyo número es infinito)1 están abiertas día y noche a los hombres y también a los animales. Que entre el que quiera. No hallará pompas mujeriles aqui ni el bizarro aparato de los palacios, pero sí la quietud y la soledad. Asimismo hallará una casa como no hay otra en la faz de la Tierra. (Mienten los que declaran que en Egipto hay una parecida.) Hasta mis detractores admiten que no hay un solo mueble en la casa. Otra especie ridícula es que yo, Asterión, soy un prisionero. ¿Repetiré que no hay una puerta cerrada, añadiré que no hay una cerradura? Por lo demás, algún atardecer he pisado la calle; si antes de la noche volví, lo hice por el temor que me infundieron las caras de la plebe, caras descoloridas y aplanadas, como la mano abierta. Ya se había puesto el Sol, pero el desvalido llanto de un niño y las toscas plegarias de la grey dijeron que me habían reconocido. La gente oraba, huía, se prosternaba; unos se encaramaban al estilóbato del templo de las Hachas, otros juntaban piedras. Alguno, creo, se ocultó bajo el mar. No en vano fue una reina mi madre; no puedo confundirme con el vulgo; aunque mi modestia lo quiera. El hecho es que soy único. No me interesa lo que un hombre pueda trasmitir a otros hombres; como el filósofo, pienso que nada es comunicable por el arte de la escritura. Las enojosas y triviales minucias no tienen cabida en mi espíritu, que está capacitado para lo grande; jamás he retenido la diferencia entre una letra y otra. Cierta impaciencia generosa no ha consentido que yo aprendiera a leer. A veces lo deploro porque las noches y los días son largos. Claro que no me faltan distracciones. Semejante al carnero que va a embestir, corro por las galerías de piedra hasta rodar al suelo, mareado. Me agazapo a la sombra de un aljibe o a la vuelta de un corredor y juego a que me buscan. Hay azoteas desde las que me dejo caer, hasta ensangrentarme. A cualquier hora puedo jugar a estar dormido, con los ojos cerrados y la respiración poderosa. (A veces me duermo realmente, a veces ha cambiado el color del día cuando he abierto los ojos). Pero de tantos juegos el que prefiero es el de otro Asterión. Finjo que viene a visitarme y que yo le muestro la casa. Con grandes reverencias le digo: Ahora volvemos a la encrucijada anterior o Ahora desembocamos en otro patio o Bien decía yo que te gustaría la canaleta oAhora verás una cisterna que se llenó de arena o Ya veras cómo el sótano se bifurca. A veces me equivoco y nos reímos buenamente los dos. No sólo he imaginado esos juegos; también he meditado sobre la casa. Todas las partes de la casa están muchas veces, cualquier lugar es otro lugar. No hay un aljibe, un patio, un abrevadero, un pesebre; son catorce (son infinitos) los pesebres, abrevaderos, patios, aljibes. La casa es del tamaño del mundo; mejor dicho, es el mundo. Sin embargo, a fuerza de fatigar patios con un aljibe y polvorientas galerías de piedra gris he alcanzado la calle y he visto el templo de las Hachas y el mar. Eso no lo entendí hasta que una visión de la noche me reveló que también son catorce (son infinitos) los mares y los templos. Todo está muchas veces, catorce veces, pero dos cosas hay en el mundo que parecen estar una sola vez: arriba, el intrincado Sol; abajo, Asterión. Quizá yo he creado las estrellas y el Sol y la enorme casa, pero ya no me acuerdo. Cada nueve años entran en la casa nueve hombres para que yo los libere de todo mal. Oigo sus pasos o su voz en el fondo de las galerías de piedra y corro alegremente a buscarlos. La ceremonia dura pocos minutos. Uno tras otro caen sin que yo me ensangriente las manos. Donde cayeron, quedan, y los cadáveres ayudan a distinguir una galería de las otras. Ignoro quiénes son, pero sé que uno de ellos profetizó, en la hora de su muerte, que, alguna vez llegaría mi redentor. Desde entonces no me duele la soledad, porque sé que vive mi redentor y al fin se levantará sobre el polvo. Si mi oído alcanzara todos los rumores del mundo, yo percibiría sus pasos. Ojalá me lleve a un lugar con menos galerías y menos puertas. ¿Cómo será mi redentor?, me pregunto. ¿Será un toro o un hombre? ¿Será tal vez un toro con cara de hombre? ¿O será como yo? El Sol de la mañana reverberó en la espada de bronce. Ya no quedaba ni un vestigio de sangre. -¿Lo creerás, Ariadna? -dijo Teseo-. El minotauro apenas se defendió. 1. El original dice catorce, pero sobran motivos para inferir que en boca de Asterión, ese adjetivo numeral vale por infinitos. FIN El Aleph, 1944"
resultado = split_sentences(texto_ejemplo)

for i, oracion in enumerate(resultado, 1):
    print(f"Oración {i}: {oracion}\n")
    print("--")
