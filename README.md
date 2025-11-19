Si bien el motivo de este proyecto es aprender sobre redes transformers, hay que mencionar que se incorpora nuevamente Seq2seq.
Y es que al fin y al cabo debemos pensar a Seq2seq no solo como un tipo de red neuronal sino tambien como un patrón de diseño aplicable a otras redes.
Si vemos que en Seq2seq tenemos un encoder y un decoder, podemos integrar diferentes tipos de redes a cada uno, o obtener su output general y trabajarlo en otra red.

Este proyecto tambien incorpora NLP (Natural Language Processing: Procesamiento del lenguaje natural), lo que nos acerca a tecnologías conocidas como ChatGPT.
Ya quería llegar a este tipo de tecnologías. Basicamente a una entrada, se nos respondera con un texto generado por el aprendizaje neuronal.
Es decir que si dieramos dos veces exactamente la misma entrada, el modelo NLP, aunque en esencia responda lo mismo, lo hará con un lenguaje natural y no dando exáctamente las mismas palabras sin cambio alguno.

    "Las aves normalmente cantan de dia" o "Las aves suelen cantar en el dia"

Ambas frases tienen el mismo significado, pero no son exactamente las mismas palabras.
De el mismo modo, la salida podra ser la misma, pero como lo diga el lenguaje no. Pues NLP esta generando la frase al momento y no simplemente repitiendola ante un estimulo.
Esta tecnología se usó, si mal no recuerdo, hasta chatGPT 4. Aunque claro, nosotros la usaremos de un modo basico. Con un dataset limitado y el proyecto se basara en intentar generar a partir de un titular, un resumen de ese titular.

    "Titular": "El servicio meteorológico emitió una alerta por vientos fuertes para la costa atlántica.",
    "resumen": "Alertan por vientos fuertes en la costa."

No se si lo mencioné antes, pero el modo que tengo de estudiar mi roadmap en redes neuronales, es simplemente solicitar el codigo a chatgpt, y simplemente analizarlo hasta el máximo, hasta que ya no me queden dudas. Explicarmelo a mi mismo en un texto (Para este caso el README.md) es otra forma de estudio que suelo usar. Pero cuando aprender programación no fue leer codigo ajeno? 

Al momento la red Transformer que pretende realizar los resumenes, funciona. Genera texto en lenguje natural, y breve. Pero falla en dar un resumen coherente y acorde al texto que debe resumir.
Quiero decir con esto, que podriamos ya estudiar la red asi como está. Pues aunque intento realizar modificaciones que solucionen la incoherencia de las salidas (soluciones que a continuacion mencionaré pues son utiles para enfocar como ver un problema relativo a este tipo de trabajos), quizas sea el problema usar datasets del tamaño adecuado para un solo programador, con una sola notebook. Aun así se podra estudiar teniendo en cuenta esa salvedad.

En la primer ejecución de la red, obtuve esto:

![alt text](miscellaneous/1.jpeg)

Un espanto, pero en esta ocación el error estaba en que el modelo esperaba inputs ya tokenizados. Es decir como valores enteros en un batch. Pero en cambio recibia texto.

Corregido ese error obtuve:

![alt text](miscellaneous/2.jpeg)

Donde claramente se ve una mejora. Aunque no tenía relacion con el texto a resumir ni tanta coherencia.
El cambio que me sugería ChatGPT era implementar Attention al modelo LSTM que se usa en seq2seq. Pero yo preferí tomar otro camino y enfocarme en el Transformer. Guarde los pesos entrenados y a cada ejecución siguiente del codigo, se reentrenarían sobre esos mismos pesos ya entrenados. Obtuve esto:

![alt text](miscellaneous/3.jpeg)

Volví a re-entrenar y obtuve:

![alt text](miscellaneous/4.jpeg)

Y ya ahi vemos como no solo tiene coherencia, sino que realizó correctamente el resumen de algunos titulares, mientras que en otros se acercó al menos. Una belleza.

En definitiva guardar los pesos guardados, para re-entrenar sobre esos pesos y no una matriz de valores randoms cada vez, es un metodo a tener en cuenta no solo para este modelo. Sino para cualquier tipo de modelo donde trabajemos con estados H o transformaciones lineales de cualquier tipo.

Hasta que punto podriamos mejorar el trabajo de esta red? Re-entrenando pesos, no lo se. Otras soluciones como incorporar Attentión a LSTM tambien o expandir el dataset, serían significativas. Pero no es algo en lo que nos vamos a ocupar aquí. Considero que para lo que es aprender que es un Transformer, la red ya esta operando a un nivel decente.

Antes de seguir admiremos el progreso que tuvo la red entre entrenamiento y entrenamiento. A mi personalmente me recordó un viejo pensamiento que tuve incluso antes de conocer ChatGPT. El lenguaje humano, separado de entendimientos mas profundos (que abre caminos a usos del lenguaje como la poesía o lenguajes logicos), puede ser llevado en su forma basica de estimulo, respuesta, a algebras y estadisticas. Verlo llevado a un codigo que lo demuestra no deja de impactarme. Que un niño aprenda que ante la palabra "hambre" puede recibir comida, es mas o menos lo que esta red realiza (pensamientos de alguien que viaja en colectivo :P ). Con más recursos podriamos crear nuestro ChatGPT.
Bueno, aun no me hecho flores, vamos a ver como funciona esto...

<h2>Estructura</h2>

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SRC_LEN = 128
MAX_TGT_LEN = 40
BATCH_SIZE = 16
EPOCHS = 12
LR = 3e-4
D_MODEL = 128
NHEAD = 4
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DIM_FF = 512
DROPOUT = 0.1
MODEL_PATH = "transformer_summarizer.pt"

<h2>Main</h2>

Una vez escuche decir a alguien de sistemas que lo mejor para aprender una nueva tecnología, es ver que problema vino a solucionar la misma. Y es sabio pensar asi, porque ninguna tecnología se abre paso si no es realmente necesitada.
RNN, LSTM, Seq2seq tienen una memoria, si, pero comparada con Transformer, el paso fue gigante y permitió que secuencias mas largas, como por ejemplo un texto mas largo que "hola mundo", fueran comprendidas por la red neuronal en cuestion.
Y con comprendidas me refiero a que al igual que ChatGPT, si dieramos un texto de entrada con varios contextos, sujetos, verbos, Transformer sabria exactamente a que prestar más atención.
Mirando la frase:

    "Una familia de Cordoba encontró un capibara viviendo en su casa, en la noche en que llegaban de sus vacaciones en Mexico"

Vemos que es una secuencia de palabras que involucran mas que solo un "Hola mundo, aprendo redes neuronales".
Tenemos un sujeto principal "familia" otro secundario, "capibara", tenemos el verbo que es nucleo del enunciado, "encontró" etc.

Recordemos que en una secuencia tenemos una referencia de tiempo t, donde cada palabra tiene un indice sub t:

    t_0: "Una", t_1: " ", t_2: "familia", t_3: " ", t_4: "de"...

Por lo tanto al llegar a t_43 (si no conté mal), la memoría, que por ejemplo en LSTM manejabamos por contexto, habria perdido la información de t_2: "familia" y justamente para lo que importa al entendimiento de la frase, el sujeto principal es importantisimo.
De modo que Transoformer viene a solucionar esto, pero con la solución vino una potenciasión de NLP.
Transformer incorpora la atención como una parte de la arquitectura dedicada. Y no solo una elemento secundario.

Una aclaración para lo que es a este codigo. La manera en que todo esta dentro de un archivo main.py, con 259 lineas de codigo apiladas, con todas las medidas de estructuras, clases y funciones en un solo sheet; no es ni debería ser para nada la manera de trabajar.
Lo normal es desacoplar, y modularizar el codigo. Pero para lo que es el estudio, y seguir el hilo de los datos, que todo este en la misma hoja, me sirve para la funcionalidad de VSCode de dar click a una variable y mostrarme donde mas esta presente en la hoja.

Por donde empezar con este codigo... 259 lineas no es nada facil de leer... obviamente por donde lo levante la ejecucion.

    if __name__ == "__main__":
        main()

No voy a explicar Python en este resumen. Pero solo dire simplificando mucho, que __name__ se torna un autoejecutable para el cmd. Por lo tanto con lo que empezará la ejecución del codigo será main()
Y si vamos a la funcion main, vemos que comienza con:

    with open("dataset.json", "r", encoding="utf8") as f:
        data = json.load(f)

    split = int(0.8 * len(data))
    train_items = data[:split]
    test_items = data[split:]

Vemos que abre el dataset.json en modo lectura y codificación utf8, lo renombra a f y lo carga a la variable data.
La funcion splint divide data en 80%/20%, guarda el 80% en train_items y el 20% en test_items. Esto ya lo vimos en la red CNN, donde nos era conveniente testear bien el entrenamiento con datos de prueba. De esa manera separamos de nuestro dataset 80% del mismo para entrenar la red y 20% para testearla.

Sigue:

    src_texts = [x["text"] for x in train_items]
    tgt_texts = [x["summary"] for x in train_items]

No tengo que investigar esta sintaxis para ver que recorre con un for el train_items y separa en src_texts los textos y en tgt_texts los summary. Y en este punto me doy cuenta de que no miramos nuestro dataset.

    [
        {
            "text": "La policía encontró un perro perdido en la costa de Mar del Plata.",
            "summary": "Hallaron un perro perdido."
        },
        {
            "text": "El dólar blue subió 30 pesos en la apertura del mercado.",
            "summary": "El blue volvió a subir."
        },
        {
            "text": "Una tormenta eléctrica provocó cortes de luz en varios barrios de Córdoba.",
            "summary": "Cortes de luz por tormenta en Córdoba."
        },
        ...
    ]

Cada elemento del dataset tiene un texto y su resumen llamado summary. src_texts guardara los textos y tgt_texts los summarys.

Pero luego tenemos esto:

    src_stoi, src_itos = build_vocab(src_texts)
    tgt_stoi, tgt_itos = build_vocab(tgt_texts)

Se llama dos veces la función build_vocab() y recibe src_texts en el primer call y tgt_texts en el segundo call. Lo retornado se guarda en dos variables cada vez. Por lo que tenemos que ir a ver que hace build_vocab.

    def build_vocab(texts, max_size=20000, min_freq=1):
        counter = Counter()
        for t in texts:
            counter.update(tokenize(t))

Recibe un parametro y define dos mas. max_size y min_freq. Inicializa un contador Counter() que se ocupa de contar elementos de tal modo que si tuvieramos por ejemplo [arbol, ave, arbol, perro] y contaramos con Counter(), obtendriamos [arbol: 2, ave: 1, perro: 1].
El bucle "for t in texts" que le sigue invoca como parametro del counter, a tokenize(t). Vamos ahi:

    def tokenize(text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9áéíóúüñ \t]+", "", text)
        return text.strip().split()

Miremoslo del modo en que si a buid_vocab() se le hubiera pasado src_texts, tokenize() estaria recibiendo los textos y no los resumenes. "for t in texts" entonces estaría recorriendo y pasando cada text a tokenize().
Tokenize() estaría pasando a minuscula cada caracter con text.lower(). re.sub trabaja de la siguiente manera.

    re.sub(patrón, reemplazo, texto)

declara un conjunto patrón que debe buscar en la cadena, si lo encuentra, lo remplaza con lo declarado en reemplazo, y texto es el string donde buscará. Si miramos nuestro patrón tenemos.

    r"[^a-z0-9áéíóúüñ \t]+

Dentro de [] el conjunto permitido, pero la presencia de ^ en el inicio nos indica una negación.
Es decir que remplazará todo lo que NO sea "a-z0-9áéíóúüñ \t". y si miramos el reemplazo, es "" nada... Es decir que eliminará todo lo que no sea lo declarado en el patrón. El simbolo + indica que se permite el remplazo de uno o mas caracteres seguidos.

Si nuestro texto fuera "Estudiantes mendocinos desarrollan una app contra plagas", tokenize(text) retorna

    ['estudiantes', 'mendocinos', 'desarrollaron', 'una', 'app', 'contra', 'plagas']

Siguiendo con la funcion build_vocab() sigue:

        items = [w for w, c in counter.most_common() if c >= min_freq]

Python tiene una forma abreviada de escribir un list comprehension: for (w, c) in counter.most_common()
Lo que hace en realidad es:

    result = []
    for (w, c) in counter.most_common():
        if c >= min_freq:
            result.append(w)

De manera que lo que estamos haciendo es un conteo de las palabras mas usadas, si es que estan en uso (c >= min_freq). Obtendiamos algo asi:

    [("los", 15), ("ciudad", 9), ("estudiantes", 1)]

Luego en el codigo:

    items = items[: max_size - 4]
    itos = ["<pad>", "<unk>", "<sos>", "<eos>"] + items

divido items en maximos de max_size (20000) - 4. Porque -4? porque al inicio de itos tengo 4 tokens ["<pad>", "<unk>", "<sos>", "<eos>"] + items. De ese modo itos queda tambien de tamaño max_size.

    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos

Lo siguiente es la construcción de un diccionario numerado llamado stoi con los elementos de itos

    itos: ["<pad>", "<unk>", "<sos>", "<eos>", "los", "ciudad", ...]
        construye:
    stoi: {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3, "los": 4, "ciudad": 5, ... }

Finalmente build_vocab() retorna itos y stoi.

    return stoi, itos

Recordemos que veniamos de mirar el main() y estabamos en:

    src_stoi, src_itos = build_vocab(src_texts)
    tgt_stoi, tgt_itos = build_vocab(tgt_texts)

