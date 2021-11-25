
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

### 2.  Feature Engineering

Essa abordagem é mais exploratória e seu objetivo principal é tentar extrair informações de dados brutos para que seja possível pelo menos criar alguns rótulos e treinar um modelo de classificação.

Algumas perguntas que essa abordagem tenta responder são:
1.  **Existe em alguma parte do texto que é dado o diagnóstico do laudo?**
2.  **Existe alguma frase padrão que indica que os exames não apresentam nada anormal?**
3.  **É possível rotular os dados com as informações apresentadas nos laudos?**

Primeiramente, o texto é pré-processado para que seja mais fácil identificar alguns padrões.  Depois foi identificado que no texto existem "**campos**" onde logo após sua ocorrência o diagnóstico é descrito. Esses campos são:

* opinião diagnóstica
* impressão diagnóstica
* hipóteses diagnósticas
* diagnósticos diferenciais
* possibilidades diagnósticas
* sugere o diagnóstico de

**Exemplo:**
![enter image description here](https://i.imgur.com/mRLIBfP.jpg)

Após o "**campo**" **opnião diagnóstica** temos o possível diagnóstico. Apenas **1140** laudos possuem esse **campo** no seu texto.  Usando regex o possível diagnóstico é extraído. Porém, mesmo com a extração desse campo, não temos um padrão seguido por quem escreveu, então o diagnóstico de pneumonia intersticial pode ser descrito de várias maneiras, impossibilitando o uso dele como rótulo. Felizmente, agora com o possível diagnóstico mais filtrado, é possível identificar novos padrões. Explorando o dado, foi encontrado algumas frases importantes.

#### Frases de diagnósticos  **normais**:

-   ausência de lesões traumáticas do parênquima pulmonar ao presente método de estudo
-   estudo radiográfico do tórax sem alterações significativas
-   aspecto radiológico normal do tórax
-   estudo tomográfico do tórax dentro dos padrões da normalidade
-   ausência de lesões traumáticas

#### Frases de diagnósticos que indicam alguma  **patologia**:

-   pneumopatia inflamatória lesões fibroatelectásicas no ápice do pulmão esquerdo
-   tromboembolismo pulmonar agudo bilateral

Alguns laudos podem apresentar  **mais de uma**  patologia como possível diagnóstico.
É mais fácil rotular o dado que apresenta diagnósticos **normais**, porque seu texto é mais "comportado".  Então, verificando se no texto se existem frases que indicam normalidade, o dado é rotulado como NORMAL e o restante como PATOLOGIA. No final o conjunto de dados fica assim:

![enter image description here](https://i.imgur.com/aCCNYYh.png)

Esse dado é apenas os 1140 laudos que possuem o campo de diagnósticos. Então vamos usar ele para treinar um modelo e inferir sobre o restante do dado. Os modelos utilizados para classificar foram:
#### **Naive Bayes**
![enter image description here](https://i.imgur.com/puAXjkV.png)

#### **SGDClassifier**

![enter image description here](https://i.imgur.com/ZQipGF1.png)

Não precisou testar muitos modelos, pois os primeiros já deram resultados bastante bons. É possível verificar que eles conseguiram entender de acordo com o dado passou os exames que podem ser NORMAIS ou que podem possuir alguma PATOLOGIA.

**Conclusão**: Mesmo usando os campos do próprio texto, o dado ainda é feito de maneira sintética. Caso existe um rótulo real para esses laudos, seria interessante fazer uma validação e verificar se de fato o modelo conseguiu identificar resultados normais de patológicos.

### 3.  Extração de Entidades Nomeadas usando BioBERTpt
O BioBERTpt - Portuguese Clinical and Biomedical BERT é um modelo baseado no BERT para língua portuguesa e treinado em notas clínicas e literatura biomédica.

Link: https://huggingface.co/pucpr

O objeitvo desse notebook é extrair entidades que possam auxiliar na separação de diagnósticos dos laudos de exames radiológicos. Foram usados 5 modelos pré-treinados do BioBERTpt: Diagnostic, Disease, Sign, Disorder e Finding.

O algoritmo executa os seguintes passos:

    1. Itera sobre cada modelo (são modelos carregados separadamente)
    2. Codifica o texto de acordo com BioBERTpt
    3. Gera os input_ids de cada palavra dos textos codificados
    4. Gera a label (Entidade) para cada input_id 
    5. Mapeia a label para cada palavra (Ex: 'enfisema': 'B-Disorder') de acordo com o nome dado pelo modelo
    6. Monta um Dataframe indicando o Diagnostic, Disease, Sign, Disorder e Finding para cada laudo do conjunto de dados
   
O resultado final da extração:

![enter image description here](https://i.imgur.com/h5Mq472.png)

   **Conclusão**: De fato o uso do modelo pré-treinado conseguiu identificar bem mais os diagnósticos do que o modelo LDA, como por exemplo no documento com o docid **375232**, foi identificado a entidade Disorder para as palavras '**alterações, crônicas, de, processo, granulomatoso, sinais, sequelas, derrames, pleurais, lesões, massas, alterações, crônicas, de, processo, granulomatoso, sinais, sequelas, hepatopatia**' para um sistema que pode utilizar essas palavras para fazer triagem já é um grande começo. Porém, ele ainda tem dificuldade de encontrar entidades do tipo Sign e Disease. A entidade Disorder também não aparece em alguns documentos e em alguns as palavras extraídas não são muito significantes como por exemplo a palavra: **massa**. O potencial pra esse tipo de abordagem é promissora, com um dado anotado seria possível fazer um novo modelo que se especializa em textos radiológicos e consequentemente vai melhorar bastante a extração das entidades.

### 4. Masked language modeling (MLM) para textos radiológicos
Esse modelo é só uma coisa **extra**, porque queria verificar como funcionaria se eu treinasse um modelo para entender representações de textos radiológicos. As métricas dele foram:

![enter image description here](https://i.imgur.com/CfEp0vX.png)

Dá pra melhorar bastante, ele ainda não pareceu performar com toda capacidade, mas é preciso de uma máquina melhor porque ele consome muitos recursos.
