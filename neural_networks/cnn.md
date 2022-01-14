# Redes Convolucionais

## Prof. Saulo Oliveira [saulo.oliveira@ifce.edu.br](mailto:saulo.oliveira@ifce.edu.br)

<style>
  mark {background-color: lightgreen; padding: 0px 5px;}
</style>



### Agenda

1. Visão Computacional
2. Campos Receptivos Locais
3. Compartilhamento de Parâmetros
4. Redes Convolucionais
   1. Arquitetura
   2. Camada de convolução
   3. Camada de agrupamento (_pooling_)
   4. Camada densa

</br>

### Visão Computacional

<div style="page-break-after: always;"></div>
Visão computacional tem como objetivo final usar computadores para emular a visão humana, incluindo aprender e ser capaz de fazer inferências e realizar ações com base em entradas (estímulos) visuais. Essa área em si é um ramo da inteligência artificial (IA) cujo objetivo é emular a inteligência humana.

[^1]: Rafael C. Gonzalez and Richard E. Wood. Digital Image Processing. 4th Edition.

 . 

### Campos Receptivos Locais e Compartilhamento de Pesos

Essa região do tensor de entrada é chamada de **campo receptivo local** para o neurônio oculto. <mark>É uma pequena janela nos pixels de entrada</mark>. Cada conexão aprende um peso e o neurônio oculto também aprende um viés (*bias*) geral. Assim, pode-se deduzir que esse neurônio oculto em particular aprende a analisar seu campo receptivo local específico.

Em seguida,  o campo receptivo local é deslizado por toda a imagem de entrada. 

![]()

<mark>Uma grande vantagem do compartilhamento de pesos e vieses é que ele reduz bastante o número de parâmetros envolvidos em uma rede convolucional</mark>.

### Tensores

Tensores são as estruturas de dados mais básicas nos _Frameworks_ de Aprendizagem Profunda. <mark style="background-color: lightgreen; padding: 0px 5px">Um tensor é uma matriz multidimensional</mark> e todos os dados são encapsulados em tensores.

<img src="/Users/sauloafoliveira/github/ppgcc-classes/neural_networks/tensor.png" alt="tensor, " style="zoom:33%;" />



### Arquitetura

**Figura 1**. Uma CNN contendo todos os elementos básicos de uma arquitetura LeNet[^1]. Os últimos mapas de atributos agrupados são vetorizados e servem como entrada para uma rede neural totalmente conectada (uma _Multilayer Perceptron_). A classe à qual a imagem de entrada pertence é determinada pelo neurônio de saída com o valor mais alto.

![cnn_arch](/Users/sauloafoliveira/github/ppgcc-classes/neural_networks/cnn_arch.png)

Fonte: [Gonzalez, R. C., Woods, R. E. *Digital image processing*, 2018](https://books.google.com.br/books?id=XmZvtAEACAAJ).

[^1]:[Backpropagation Applied to Handwritten Zip Code Recognition](https://doi.org/10.1162/neco.1989.1.4.541)



**Figura 2.** Ilustração gráfica de uma passagem direta pela CNN treinada. O objetivo era reconhecer uma imagem de entrada do conjunto. Como mostra a saída, a imagem foi reconhecida corretamente como pertencente à classe 1, a classe dos aviões.

<img src="/Users/sauloafoliveira/github/ppgcc-classes/neural_networks/airplane.png" style="zoom:33%;" />

Fonte: [Gonzalez, R. C., Woods, R. E. *Digital image processing*, 2018](https://books.google.com.br/books?id=XmZvtAEACAAJ).

### Pooling (Agrupamento/Subamostragem)

<mark>O *pooling* combina unidades próximas para reduzir o tamanho da entrada na próxima camada, reduzindo as dimensões do tensor</mark>. O *pooling* comum inclui o *pooling* máximo e o *pooling* médio. Quando o *pooling* de máximo é usado, o valor máximo em uma pequena área quadrada é selecionado como o representante dessa área, enquanto o valor médio é selecionado como o representante quando o pooling médio é usado.

O resultado do uso de uma camada de *pooling* e da criação de mapas de atributos <u>amostrados ou agrupados</u> é uma versão resumida dos atributos detectados na entrada. Isso é útil, pois pequenas mudanças na localização de atributos na entrada detectada pela camada convolucional resultarão em um mapa de atributos menor com os atributos no mesmo local. <mark>Essa capacidade adicionada pelo *pooling* é chamada de invariância do modelo para a translação local</mark>.



### Código-fonte

```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

print(model.summary())
```

### Referências

- GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. Deep learning. Cambridge, USA: MIT Press, 2016. (Adaptive Computation and Machine Learning series). ISBN 9780262035613.
- Fei-Fei Li et al. CS231n: Convolutional Neural Networks for Visual Recognition. http://cs231n.stanford.edu/. 2020, Accessed on Feb 2021.
- Gonzalez, R. C., Woods, R. E. Digital image processing, USA: Pearson, 2018. ISBN 9781292223049.
- Juan Cruz Martinez. Introduction to Convolutional Neural Networks CNNs. https://aigents.co/blog/publication/introduction-to-convolutional-neural-networks-cnns. 2020, Accessed on Feb 2021.
- HUAWEI. Deep Learning Overview. 2020, Accessed on Feb 2021.
