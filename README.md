# 🏡 Previsão de Preços do Airbnb (Rio de Janeiro) — Machine Learning End-to-End

Este projeto desenvolve uma solução completa de **Ciência de Dados e Engenharia de Machine Learning**, desde a ingestão e consolidação de cerca de **900 mil registros** até o treinamento de modelos preditivos e o deploy de uma aplicação web interativa em **Streamlit**.

O objetivo de negócio é fornecer uma ferramenta capaz de estimar o valor justo da diária de um imóvel no Rio de Janeiro com base em suas características físicas, regras de locação, comodidades e localização geográfica (latitude/longitude), auxiliando tanto proprietários comuns na precificação quanto locatários na identificação de oportunidades.

---

## 🛠️ Tecnologias e Ferramentas Utilizadas

* **Linguagem:** Python 3.12
* **Engenharia e Manipulação de Dados:** `pandas`, `numpy`, `pathlib`
* **Análise Exploratória (EDA) e Visualização:** `seaborn`, `matplotlib`, `plotly.express`
* **Machine Learning:** `scikit-learn` (`LinearRegression`, `RandomForestRegressor`, `ExtraTreesRegressor`), `joblib`
* **Deploy / Front-end Analítico:** `streamlit`

---

## ⚙️ Arquitetura do Pipeline (Etapas do Projeto)

### 1. Ingestão em Lote e Otimização de Memória (ETL)
* **Consolidação Sazonal:** Leitura automatizada de múltiplos arquivos CSV mensais (abril/2018 a maio/2020) via `pathlib`, extraindo metadados de `mes` e `ano` diretamente do nome dos arquivos para capturar o efeito da sazonalidade nos preços.
* **Performance de Ingestão:** Uso de listas em memória antes da execução única do `pd.concat()`, evitando a realocação repetitiva de memória RAM durante o loop de leitura.
* **Downcasting de Tipos:** Limpeza de strings monetárias (`$` e `,`) e conversão direta para `np.float32`, reduzindo pela metade o consumo de memória das variáveis contínuas críticas (`price` e `extra_people`).

### 2. Data Wrangling e Tratamento de Outliers
* **Redução de Dimensionalidade Inicial:** Filtragem qualitativa de **108 para 34 colunas**, eliminando URLs, identificadores únicos, campos de texto livre e variáveis redundantes.
* **Tratamento de Dados Faltantes (`NaN`):** Exclusão automatizada de colunas com mais de 300.000 registros nulos e remoção residual de linhas incompletas (`dropna`), preservando uma base sólida de ~897 mil registros.
* **Remoção Estatística de Anomalias (Método de Tukey / IQR):** Implementação de funções modulares baseadas na Amplitude Interquartil ($Q_3 - Q_1$) para filtrar *outliers* extremos ($< Q_1 - 1.5 \times \text{IQR}$ ou $> Q_3 + 1.5 \times \text{IQR}$), focando o aprendizado do modelo na realidade de imóveis padrão do mercado.

### 3. Feature Engineering e Encoding
* **Quantificação de Comodidades (`amenities`):** Transformação de listas textuais complexas de itens do imóvel em uma variável numérica discreta (quantidade total de comodidades), reduzindo ruído e alta cardinalidade.
* **Agrupamento de Categorias Raras:** Consolidação de tipos de imóveis (`property_type`) de baixa frequência na categoria `"Outros"`.
* **Codificação Vetorizada:** Conversão de variáveis binárias (`t`/`f`) em `1` e `0` via `np.where()` e aplicação de *One-Hot Encoding* (`pd.get_dummies`) para variáveis categóricas nominais (`room_type`, `bed_type`, `cancellation_policy`, `property_type`).

### 4. Modelagem Preditiva e Avaliação
Divisão da base em conjuntos de **Treino (80%)** e **Teste (20%)** via `train_test_split` e avaliação comparativa de três algoritmos utilizando o **Coeficiente de Determinação ($R^2$)** e o **Erro Quadrático Médio ($RMSE$)**:
1. **Linear Regression:** Modelo base (*baseline*) para avaliação de relações lineares.
2. **Random Forest Regressor:** Modelo de *ensemble* baseado em *bagging* de árvores de decisão.
3. **Extra Trees Regressor (Modelo Escolhido):** Apresentou a melhor relação entre capacidade preditiva ($R^2 > 95\%$) e generalização espacial.

* **Feature Importance:** Análise do peso das variáveis no modelo vencedor (`feature_importances_`), permitindo a exclusão de atributos irrelevantes (como `is_business_travel_ready`) para otimizar o modelo final antes do deploy.

### 5. Deploy com Streamlit
Empacotamento do modelo otimizado (`joblib`) e construção de uma interface interativa onde o usuário preenche as características do imóvel e recebe instantaneamente a previsão do valor da diária.

## 6. 📚 Créditos e Referências
Desenvolvimento e Metodologia: Projeto prático desenvolvido durante a formação Python Impressionador da Hashtag Treinamentos, com adaptações e otimizações de código para compatibilidade com Pandas 2.0+/3.0 e gerenciamento eficiente de memória.

Fonte de Dados e Inspiração: Conjunto de dados disponibilizado publicamente no Kaggle por Allan Bruno.

---

## 📂 Estrutura do Repositório

```text
├── dataset/                       # Diretório para os CSVs mensais brutos (ignorado no Git)
├── previsao_Airbnb_Rio.ipynb      # Notebook completo (ETL, EDA, Outliers e Treinamento)
├── deploy_projeto_airbnb.py       # Aplicação web interativa construída em Streamlit
├── .gitignore                     # Bloqueio de bases brutas (.csv) e modelos pesados (.joblib)
└── README.md                      # Documentação técnica do projeto




