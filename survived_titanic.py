# Importanto as bibliotecas que serão usadas no projeto
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple

# vamor criar a classe para e o método de remoção de colunas que contenha uma 
# alta perda de dados!


class DataPrep:
    def __init__(self, data: pd.DataFrame) -> None:
        # Inicializa a classe DataPrep com a base de dados do Titanic
        self.data = data

    def remover_colunas(self) -> None:
        # Remove as variáveis que não serão utilizadas pelo modelo
        remocao_colunas = [
            'PassengerId',
            'Ticket',
            'Name',
            'Cabin',  # variável com muitos dados faltantes
            'Embarked',
            'SibSp',
            'Parch'
        ]
        self.data.drop(columns=remocao_colunas, inplace=True)

    # Iremos fazer o tratamento de dados que são nulos
    def tratar_dados_nulos(self) -> None:
        "Faz o tratamento das variáveis nulas"
        # Imputar mediana das idades por classe e sexo
        self.data['Age'] = self.data.groupby(['Pclass', 'Sex'])['Age'] \
            .apply(lambda x: x.fillna(x.median()))
        
        # Imputar local de embarque
        self.data['Embarked'] = self.data['Embarked'].fillna('S')

    # Vamor tratar das variáveis de categorias, as transformando em true ou
    # false (bool)
    def tratar_variaveis_categorias(self) -> None:
        "Faz o tratamento das variáveis categóricas"
        # Label Encoding da variável Sex
        sexo = {'male': 0, 'female': 1}
        self.data['Sex'] = self.data['Sex'].map(sexo)
        embarked_dummies = pd.get_dummies(self.data['Embarked'],
                                          prefix='Embarked')

        # One Hot Encoding da variável Embarked
        embarked_dummies = pd.get_dummies(self.data['Embarked'])
        self.data = pd.concat([self.data, embarked_dummies], axis=1)

        # Remover após criar as dummies
        self.data.drop(columns=['Embarked'], inplace=True)

    # Iremos normalizar os dados utilizando o MinMaxScaler
    def normalizar_dados(self) -> None:
        variaveis = self.data.drop(columns='Survived')
        var_cols = variaveis.columns
        resposta = self.data['Survived']

        scaler = MinMaxScaler()
        variaveis = scaler.fit_transform(variaveis)
        variaveis = pd.DataFrame(variaveis, columns=var_cols)

        self.data = pd.concat([variaveis, resposta], axis=1)

    # Criaremos uma coluna que se chamará FamilySize que mostrará a quantidade
    # de famílias, de cada passaseiros, que sobreviveram.
    def criar_variaveis(self) -> None:
        self.data['FamilySize'] = self.data['SibSp'] + self.data['Parch'] + 1

    # vamos treinar o nosso algoritmo e testá-lo
    def separar_treino_teste(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        treino, teste = train_test_split(self.data, test_size=0.3,
                                         random_state=2024)
        return treino, teste

    # preparando os dados dos métodos anteriores, para treino
    def preparar_dados(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Executa todas as etapas de transformação de dados
        self.tratar_dados_nulos()
        self.criar_variaveis()
        self.tratar_variaveis_categorias()
        self.remover_colunas()
        self.normalizar_dados()

        treino, teste = self.separar_treino_teste()
        return treino, teste
