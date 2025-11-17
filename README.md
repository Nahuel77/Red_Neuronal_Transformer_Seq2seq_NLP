Si bien el motivo de este pryecto es aprender sobre redes transformers, hay que mencionar que se incorpora nuevamente Seq2seq.
Y es que al fin y al cabo debemos pensar a Seq2seq no solo como un tipo de red neuronal sino tambien como un patron de diseño aplicable a otras redes.
Si vemos que en Seq2seq tenemos un encoder y un decoder, podemos integrar diferentes tipos de redes a cada uno.

Este proyecto tambien incorpora NLP (Natural Language Processing: Procesamiento del lenguaje natural), los que nos acerca a tecnologías conocidas como ChatGPT.
Ya quería llegar a este tipo de tecnologías. Basicamente a una entrada, se nos respondera con un texto generado por el aprendizaje neuronal.
Es decir que si dieramos dos veces exactamente la misma entrada, el lenguaje, aunque en esencia responda lo mismo, lo hara con un lenguaje natural y no dando exactamente las mismas palabras sin cambio alguno.

    "Las aves normalmente cantan de dia" o "Las aves suelen cantar en el dia"

Ambas frases tienen el mismo significado, pero no son exactamente las mismas palabras.
De el mismo modo, la salida podra ser la misma, pero como lo diga el lenguaje no. Esta tecnología se usó, si mal no recuerdo, hasta chatGPT 4. Aunque claro, nosotros la usaremos de un modo basico. Con un dataset limitado y el proyecto se basara en intentar generar a partir de un titular, un resumen de ese titular.

    "Titular": "El servicio meteorológico emitió una alerta por vientos fuertes para la costa atlántica.",
    "resumen": "Alertan por vientos fuertes en la costa."

No se si lo mencioné antes, pero el modo que tengo de estudiar mi roadmap en redes neuronales, es simplemente solicitar el codigo a chatgpt, y simplemente analizarlo hasta el máximo, hasta que ya no me queden dudas. Explicarmelo a mi mismo en un texto (Para este caso el README.md) es otra forma de estudio que suelo usar.

Al momento la red Transformer que pretende realizar los resumenes, funciona. Genera texto en lenguje natural, y breve. Pero falla en dar un resumen coherente y acorde al texto que debe resumir.
Quiero decir con esto anterior, que podriamos ya estudiar la red asi como está. Pues aunque intento realizar modificaciones que solucionen la incoherencia de las salidas (soluciones que a continuacion mencionaré pues son utiles para enfocar como ver un problema relativo a este tipo de trabajos), quizas los recursos de usar datasets del tamaño adecuado para un solo programador, con una sola notebook. Aun así se podra estudiar teniendo en cuenta esa salvedad.

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

Hasta que punto podriamos mejorar el trabajo de esta red? Re-entrenando pesos, no lo se. Otras soluciones como incorporar Attentión a LSTM tambien serían significativas. Pero no es algo en lo que nos vamos a ocupar aquí. Considero que para lo que es aprender que es un Transformer, la red ya esta operando a un nivel decente.

Antes de seguir admiremos el progreso que tuvo la red entre entrenamiento y entrenamiento. A mi personalmente me recordó un viejo pensamiento que tuve incluso antes de conocer ChatGPT. El lenguaje humano, separado de entendimientos mas profundos (que abre caminos a usos del lenguaje como la poesía o lenguajes logicos), puede ser llevado en su forma basica de estimulo, respuesta, a algebras y estadisticas. Verlo llevado a un codigo que lo demuestra no deja de impactarme. Que un niño aprenda que ante la palabra "hambre" puede recibir comida, es mas o menos lo que esta red realiza (pensamientos de alguien que viaja en colectivo :P ). Con más recursos podriamos crear nuestro ChatGPT.
Bueno, aun no me hecho flores, vamos a ver como funciona esto...

<h2>Transformer</h2>

