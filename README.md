# Previsão de Sobrevivência no Titanic

Este projeto tem como objetivo prever se um passageiro sobreviveu ou não ao desastre do Titanic, utilizando aprendizado de máquina e processamento de dados com o Pandas e outras bibliotecas Python. O código implementa diversas etapas de preparação de dados, incluindo a limpeza, transformação e normalização de variáveis, além de usar o modelo de aprendizado supervisionado para fazer previsões.

## Estrutura do Código

O código é organizado em uma classe chamada `DataPrep`, que contém diversos métodos para processar os dados de entrada. Abaixo está uma explicação detalhada de cada um desses métodos.

### 1. **Classe DataPrep**

A classe `DataPrep` é responsável por preparar os dados para o treinamento de um modelo de aprendizado de máquina. Ela contém os seguintes métodos:

#### **`__init__(self, data: pd.DataFrame) -> None`**
Este é o construtor da classe. Ele recebe o conjunto de dados como entrada (um `DataFrame` do Pandas) e armazena-o como um atributo da classe.

#### **`remover_colunas(self) -> None`**
Este método remove colunas que não são úteis para a análise e que podem ter muitas informações faltantes. As colunas removidas são:
- `PassengerId`: Identificador único para cada passageiro (não é necessário para o modelo).
- `Ticket`: Número do bilhete do passageiro.
- `Name`: Nome do passageiro.
- `Cabin`: Número da cabine do passageiro.
- `Embarked`: Porto onde o passageiro embarcou.
- `SibSp`: Número de irmãos e cônjuges do passageiro a bordo.
- `Parch`: Número de pais e filhos do passageiro a bordo.

Essas colunas são removidas para simplificar os dados, já que não são essenciais para a previsão.

#### **`tratar_dados_nulos(self) -> None`**
Neste método, lidamos com valores nulos:
- A coluna `Age` (idade do passageiro) tem valores ausentes. Para preenchê-los, utilizamos o método `groupby`, que agrupa os dados por `Pclass` (classe do bilhete) e `Sex` (sexo). A mediana da idade é usada para substituir os valores ausentes em cada grupo.
- A coluna `Embarked` (porto de embarque) também tem valores ausentes e é preenchida com o valor `'S'` (Southampton).

#### **`tratar_variaveis_categorias(self) -> None`**
Aqui, as variáveis categóricas, como `Sex` e `Embarked`, são transformadas em variáveis binárias ou "dummies":
- `Sex` é mapeado para `1` (masculino) e `0` (feminino).
- A coluna `Embarked` é transformada em variáveis dummies usando a função `pd.get_dummies`. Isso cria uma nova coluna para cada categoria de embarque (`C`, `Q`, `S`), facilitando a utilização dessas variáveis no modelo.

#### **`normalizar_dados(self) -> None`**
Neste método, as variáveis numéricas são normalizadas para garantir que todas as variáveis tenham a mesma escala, utilizando o `MinMaxScaler` do `sklearn`. A normalização ajuda a melhorar a convergência de algoritmos de aprendizado de máquina e impede que variáveis com valores maiores dominem o modelo.

#### **`criar_variaveis(self) -> None`**
Aqui, criamos a variável `FamilySize`, que é a soma de `SibSp` (número de irmãos/cônjuges) e `Parch` (número de pais/filhos). Essa variável pode fornecer informações sobre o tamanho da família do passageiro e pode ter um impacto na probabilidade de sobrevivência.

#### **`separar_treino_teste(self) -> Tuple[pd.DataFrame, pd.DataFrame]`**
Este método divide o conjunto de dados em duas partes:
- **Treinamento (70%)**: Usado para treinar o modelo de aprendizado de máquina.
- **Teste (30%)**: Usado para testar a performance do modelo em dados não vistos.

A divisão é feita utilizando a função `train_test_split` do `sklearn`.

#### **`preparar_dados(self) -> Tuple[pd.DataFrame, pd.DataFrame]`**
Este é o método principal que aplica todas as etapas anteriores:
1. Trata os dados nulos.
2. Cria as variáveis adicionais.
3. Aplica a transformação de variáveis categóricas.
4. Remove as colunas desnecessárias.
5. Normaliza os dados.

Ao final, ele retorna os conjuntos de treino e teste.

## Descrição das Variáveis

O conjunto de dados contém diversas variáveis que ajudam a descrever os passageiros. Aqui estão as principais variáveis:

- **passengerId**: Identificador único para cada passageiro.
- **survived**: A variável de resposta que queremos prever. 1 = sobreviveu, 0 = não sobreviveu.
- **Pclass**: Classe do bilhete do passageiro. 1 = primeira classe, 2 = segunda classe, 3 = terceira classe. Esta variável tem grande relação com o status socioeconômico do passageiro.
- **name**: Nome do passageiro.
- **sex**: Sexo do passageiro.
- **age**: Idade do passageiro. Se a idade do passageiro for menor que 1, então a idade terá um valor fracional (ex.: 0.75). Se a idade for estimada, o valor terá o formato XX.5 (ex.: 35.5 para uma idade estimada de 35 anos).
- **SibSp**: Número de irmãos e cônjuges do passageiro que também estavam a bordo.
- **parch**: Número de pais e filhos do passageiro que também estavam a bordo.
- **ticket**: Número do bilhete do passageiro.
- **fare**: Valor da tarifa paga pelo passageiro.
- **cabin**: Número da cabine do passageiro.
- **embarked**: Porto onde o passageiro embarcou. C = Cherbourg, Q = Queenstown, S = Southampton.

## Conclusão

Este projeto foi desenvolvido para prever a sobrevivência de passageiros do Titanic com base em suas características. Ao utilizar técnicas de limpeza e transformação de dados, como a remoção de colunas desnecessárias, preenchimento de dados nulos, transformação de variáveis categóricas e normalização, criamos um conjunto de dados que pode ser utilizado em modelos de aprendizado de máquina para fazer previsões sobre a sobrevivência dos passageiros.

## Como Usar

1. Carregue os dados:
   ```python
   df = pd.read_csv("titanic.csv")

2. Instancie a classe *DataPrep*:
   ```python
   prep = DataPrep(data=df)

3. Prepare os dados:
   ```python
   treino, teste = prep.preparar_dados()

4. O conjunto *treino* estará pronto para treinar um modelo de aprendizado de máquina, e o *teste* será usado para avaliar a performance do modelo
