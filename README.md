# MLP do Zero — XOR e MNIST

Implementação de um Multi-Layer Perceptron (MLP) utilizando apenas NumPy, sem frameworks de deep learning.

## Por que dois notebooks?

O repositório contém dois notebooks distintos com papéis complementares.

**`Notebooks/MLP_XOR.ipynb`** foi desenvolvido primeiro, no ambiente Google Colab, como caderno de aprendizado. O objetivo era entender e implementar cada peça do MLP manualmente antes de escalar para um problema real: forward pass, backpropagation, curva de loss, efeito dos hiperparâmetros e refatoração para arquitetura modular. Cada teste adicionado ao notebook registra uma decisão técnica específica e a motivação por trás dela. Esse caderno é a base conceitual de tudo que veio depois.

**`Notebooks/MLP_MNIST.ipynb`** foi desenvolvido em seguida, também no Colab, aplicando os conceitos consolidados no XOR ao problema de classificação do MNIST. A arquitetura, as funções de ativação, o esquema de backpropagation e as decisões de inicialização de pesos partem diretamente do que foi estudado e testado no caderno anterior.

Essa separação é intencional: o MLP_XOR documenta o processo de aprendizado; o MLP_MNIST documenta a aplicação.

## Como rodar

```bash
pip install numpy matplotlib
```

Ambos os notebooks foram desenvolvidos no Google Colab. Para rodar localmente, basta abrir o arquivo `.ipynb` em Jupyter e executar as células em ordem. O MNIST é carregado via `keras.datasets.mnist` — o Keras é usado exclusivamente para acesso ao dataset, não para construção ou treinamento do modelo.

## Arquitetura — MNIST

A rede implementada no MLP_MNIST possui duas camadas ocultas com ativação ReLU e camada de saída com softmax. Os detalhes de cada decisão de arquitetura, incluindo a escolha do número de neurônios, da função de ativação e do esquema de inicialização de pesos, estão documentados em `Doc/Documentacao.md`.

## Estrutura do repositório

```
.
├── README.md
├── Notebooks/
│   ├── MLP_XOR.ipynb
│   └── MLP_MNIST.ipynb
├── Doc/
│   └── Documentacao.md
└── Assets/
    ├── mlp_xor_arquitetura.jpeg
    └── xor_nao_linear.jpeg
```

## Resultados

Os resultados do treinamento no MNIST, incluindo acurácia final, curva de loss e comparação entre configurações, estão registrados em `Doc/Documentacao.md`.

## Decisões e dificuldades

A seção "Decisões e dificuldades" com resposta às três perguntas obrigatórias está em `Doc/Documentacao.md`.
