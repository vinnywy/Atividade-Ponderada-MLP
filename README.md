# MLP do Zero — XOR e MNIST
 
Implementação de um Multi-Layer Perceptron (MLP) utilizando apenas NumPy, sem frameworks de deep learning. 

**Declaração de uso de IA:** Houve o uso de IA para desenvolvimento e entendimento mais profundos de cálculo e aplicação de "prints" mais complexos para visualização de resultados. Entretanto foi essencial para um entendimento melhor em processos de "como?" aplicar, sendo a motivação de decisões de aplicação um aprendizado mútuo vindo de pesquisa e teste prático.
 
## Por que dois notebooks?
 
Desenvolvi em dois notebooks com papéis complementares. O `MLP_XOR` foi feito primeiro no Colab como caderno de aprendizado, cada teste adicionado surgiu de uma dúvida real que apareceu no caminho: forward pass, backpropagation, curva de loss, efeito dos hiperparâmetros, refatoração para arquitetura modular. Esse caderno é a base de tudo que veio depois.
 
O `MLP_MNIST` foi desenvolvido em seguida, também no Colab, aplicando os conceitos consolidados no XOR ao problema de classificação do MNIST. A arquitetura, as funções de ativação, o backpropagation e a inicialização de pesos partem diretamente do que foi estudado no caderno anterior.
 
## Como rodar
 
```bash
pip install numpy matplotlib scikit-learn tensorflow
```
 
Abrir o arquivo `.ipynb` no Google Colab ou Jupyter e executar as células em ordem. O MNIST é carregado via `keras.datasets.mnist`, o Keras é usado exclusivamente para acesso ao dataset.
 
## Arquitetura — MNIST
 
```
784 (entrada) → 256 (ReLU) → 128 (ReLU) → 10 (softmax)
```
 
**Por que essa arquitetura:** segue a heurística de redução progressiva, cada camada comprime a representação antes da saída. Com 784 entradas e 10 saídas, camadas ocultas com 256 e 128 estão dentro do intervalo recomendado. Detalhes na seção 5 da documentação.
 
**Por que ReLU nas camadas ocultas:** a sigmoid tem gradiente máximo de 0.25 e satura para valores extremos, causando vanishing gradient em redes profundas. A ReLU tem derivada 1 para valores positivos, mantendo o gradiente estável.
 
**Por que Xavier:** inicialização com `uniform(0,1)` gera viés positivo que satura a sigmoid nas primeiras épocas. Xavier escala os pesos por $\sqrt{1/n_{in}}$, mantendo a variância dos gradientes estável entre camadas.
 
**Otimizadores:** SGD e Adam implementados. Adam adapta a taxa de cada peso individualmente com base no histórico de gradientes — converge mais rápido que o SGD no MNIST.
 
## Resultados
 
*(a preencher após execução no Colab)*
 
| Configuração | Otimizador | Alpha | Acurácia teste |
|---|---|---|---|
| [784, 256, 128, 10] | SGD | 0.01 | 96.84% |
| [784, 128, 64, 10] | SGD | 0.01 | 96.75% |
| [784, 256, 128, 10] | SGD | 0.05 | 97.98% |
| [784, 256, 128, 10] | Adam | 0.001 | 98.00% |
| **[784, 256, 128, 10] — Adam, 50 épocas** | **Adam** | **0.001** | **98.28%** |
 
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
 
## Decisões e dificuldades
 
**A decisão técnica mais difícil** foi a refatoração do Teste 5. Enquanto o código com variáveis nomeadas (`w11`, `w12`...) era fácil de rastrear linha por linha, a versão matricial exigiu pensar nos pesos como um objeto único por camada. Os erros de shape, `(n_saidas, n_entradas)` vs `(n_entradas, n_saidas)`, causaram erros de multiplicação matricial que levaram um tempo para depurar. A decisão de manter a refatoração foi necessária porque sem ela o MNIST seria inviável. Além de coceitos matemáticos mais avançados e funções de representação dos processos, esses necessitaram o uso de IA para um desenvolvimento mais rápido, mas foram essênciais para um entendimento mais estrutural no processo de contrução de uma rede neural.
 
**O que não funcionou:** a inicialização com `uniform(0, 1)` foi o problema mais lento de identificar. A rede treinava e convergia, então não havia erro explícito, mas a convergência era mais lenta do que deveria e algumas inicializações levavam a uma loss que ficava alta por muitas épocas antes de começar a cair. Só ao testar `uniform(-1, 1)` e depois Xavier ficou claro que a inicialização positiva estava causando saturação inicial da sigmoid.
 
**O que faria diferente:** começaria o notebook XOR já com a arquitetura baseada em matrizes e usaria a versão com variáveis individuais só como explicação nos comentários, não como código funcional. A refatoração do Teste 5 criou inconsistências de interface entre as classes, a `MLPComCurva` e a `MLPModular` têm assinaturas diferentes para `predict`, que tornaram difícil comparar os testes anteriores com os posteriores.