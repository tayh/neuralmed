
# Desafio NLP - Neuralmed

Os dados são laudos médicos derivados de tomografias, radiografias e etc. Alguns laudos descrevem se o exame encontrou alguma patologia ou não.


## Análise
O dataset possui 5000 laudos, e possuem as seguintes características:

* **'docid'**: O id pertencente a cada laudo
* **'modalidade'**: A modalidade de cada tipo de exame
* **'tipo_exame'**: o tipo de exame radiológico usado
* **'laudo_completo'**: O texto gerado pela pessoa que fez o exame

Grande parte dos dados são de modalidade CT e tipo TOMOGRAFIA. Além disso, a maioria dos dados são de exames do TÓRAX.  
![gráficos](https://i.imgur.com/rX7240x.png)

Como mostrado na nuvem de palavras que engloba o texto de todos os laudos, é possível identificar palavras no contexto radiológico como **estudo tomográfico**, **injeção ev**, **contraste** e **tomográfico computadorizado**. No contexto da anatomia humana aparecem palavras ligadas ao tórax como **aorta torácica**, **lobo superior esquerdo e direito**, **artéria pulmonar**, **coração**, etc e também algumas palavras indicando um diagnóstico, seja ele normal ou com alguma patologia: **derrame pleural**, **aspecto preservado**, **dimensões normais** e etc. 

![wordcloud](https://i.imgur.com/Xx8GSTA.png)

É possível ver que de fato o dado tem o contexto radiológico e cobre a parte do tórax.

## Abordagens:

**Problema**: Os dados possuem muitas informações importantes, porém, elas não estão rotuladas. Os diagnósticos dos laudos não são normalizados, pois estão descritos em linguagem natural, então é bem difícil quantificar ou classificar os laudos a partir dos diagnósticos.

Foram feitas duas abordagens para tentar extrair os diagnósticos do texto:

1. LDA (Linear Discriminant Analysis), onde o modelo identifica os principais tópicos dos textos. ([notebook](https://github.com/tayh/neuralmed/blob/main/An%C3%A1lise%20e%20Modelo%20LDA.ipynb))
2. Feature Engineering: Padrões no texto são encontrados e usados para extrair informações de dados brutos ([notebook](https://github.com/tayh/neuralmed/blob/main/Feature%20Engineering.ipynb))
3. Extração de Entidades Nomeadas usando NER (Named entity recognition) de modelos pré-treinados. ([notebook](https://github.com/tayh/neuralmed/blob/main/Extra%C3%A7%C3%A3o%20de%20Entidades%20com%20Biobertpt.ipynb))
4.  **Extra:**  Masked language modeling (MLM) para textos radiológicos (verificar performance e futuras aplicações) ([notebook](https://github.com/tayh/neuralmed/blob/main/Fine_tuning_MLM_para_textos_radiol%C3%B3gicos.ipynb))

### 1. LDA
O modelo LDA é bastante usado em dados de texto não rotulados, porque ele consegue identificar tópicos à partir de um conjunto de palavras. Existem várias bibliotecas que implementam o LDA. A utilizada aqui é a do gensim.

Primeiramente é preciso pré-processar o texto, removendo caracteres especiais, muitos espaços em branco e etc. Depois é aplicado bigram e trigram nas palavras para tentar dar um sentido maior ao texto e evitar muita generalização

Para o definir a quantidade de tópicos é utilizado o CoherenceModel, ele dá os valores de coerência correspondentes ao modelo LDA com o respectivo número de tópicos.

![enter image description here](https://i.imgur.com/h0fNLIh.png)

O número de tópico com o melhor valor de coerência é **6**. Então o modelo LDA foi treinado com seis tópicos.
Os tópicos gerados pelo modelo foram:

![enter image description here](https://i.imgur.com/wqLl8dq.png)

É possível ver que o modelo LDA consegue identificar tópicos que representam alguns órgãos e os aspectos que eles se encontram:

* **Tópico 0**: Coração, aspectos normais e/ou preservados
* **Tópico 1**: Pulmões (lobo esquerdo e direito) com ausência de anormalidades
* **Tópico 2**: Tórax de uma forma geral, envolvendo pulmões e a área cardiaca. Mostrando possivilmente um diagnóstico de normalidade.
* **Tópico 3**: Mediastino e contornos com aspectos preservados
* **Tópico 4**: Cúpulas diaframáticas com alterações, estruturas ósseas, alterações no mediastino e vascularização. Esse tópico parece ter pegado mais patologias do que normalidades.
* **Tópico 5**: Aorta torácica e traquéia, ambas com aspectos preservados.

**Conclusão**: Mesmo com modelo conseguir identificar tópicos por órgãos e o aspectos deles, não sei se seria útil para alguma aplicação, porque ele separa tudo de maneira muito genérica e dificilmente consegue identificar de fato um diagnóstico.
