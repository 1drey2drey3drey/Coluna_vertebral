# 🏥 Análise e Modelagem de Dados de Coluna Vertebral

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/dataset-UCI%20ML%20Repository-orange)](https://archive.ics.uci.edu/ml/datasets/Vertebral+Column)

> Projeto de Machine Learning para classificação de patologias da coluna vertebral utilizando características biomecânicas. Desenvolvido como parte do 2º Bimestre de [Nome da Disciplina].

---

## 📋 Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Dataset](#dataset)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Instalação](#instalação)
- [Como Usar](#como-usar)
- [Resultados](#resultados)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Metodologia](#metodologia)
- [Autores](#autores)
- [Licença](#licença)
- [Referências](#referências)

---

## 🎯 Sobre o Projeto

Este projeto aplica técnicas de **Análise Exploratória de Dados (EDA)** e **Machine Learning** para classificar pacientes ortopédicos em duas categorias:
- **Normal**: Sem patologias vertebrais
- **Anormal**: Com hérnia de disco ou espondilolistese

### Objetivos Principais

✅ Conduzir análise exploratória completa com tratamento de dados  
✅ Investigar relações entre variáveis biomecânicas  
✅ Implementar e comparar modelos de classificação (Naive Bayes e Regressão Logística)  
✅ Avaliar performance com métricas apropriadas (Accuracy, Precision, Recall, F1, AUC-ROC)  
✅ Otimizar modelos com validação cruzada e tuning de hiperparâmetros  

### Hipóteses de Negócio

1. **H1**: Características biomecânicas da pelve e coluna lombar são preditores significativos de patologias vertebrais
2. **H2**: Modelos de classificação podem auxiliar no diagnóstico precoce de problemas na coluna vertebral
3. **H3**: A combinação de múltiplas features biomecânicas melhora a capacidade preditiva dos modelos

---

## 📊 Dataset

### Fonte
- **Nome**: Vertebral Column Dataset
- **Origem**: UCI Machine Learning Repository
- **Kaggle**: [Vertebral Column Dataset](https://www.kaggle.com/datasets/jessanrod3/vertebralcolumndataset/data)
- **Créditos**: Dr. Henrique da Mota - Centre Médico-Chirurgical de Réadaptation des Massues, Lyon, França
- **Licença**: Database Contents License (DbCL) v1.0

### Características

| Variável | Descrição | Unidade |
|----------|-----------|---------|
| `pelvic_incidence` | Incidência pélvica | graus (°) |
| `pelvic_tilt` | Inclinação pélvica | graus (°) |
| `lumbar_lordosis_angle` | Ângulo de lordose lombar | graus (°) |
| `sacral_slope` | Inclinação sacral | graus (°) |
| `pelvic_radius` | Raio pélvico | mm |
| `degree_spondylolisthesis` | Grau de espondilolistese | - |

### Estatísticas
- **Total de observações**: 310 pacientes
- **Classes**: Normal (100) | Anormal (210)
- **Desbalanceamento**: 2.1:1
- **Features**: 6 atributos biomecânicos

---



## 🚀 Instalação

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)
- Git

### Passo a Passo

1. **Clone o repositório**

```bash
git clone https://github.com/1drey2drey3drey/Coluna_vertebral.git
cd Coluna_vertebral
```

2. **Crie um ambiente virtual** (recomendado)

```bash
# Linux/Mac
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

3. **Instale as dependências**

```bash
pip install -r requirements.txt
```

4. **Baixe o dataset**

**Opção A - Manual:**
- Acesse: https://www.kaggle.com/datasets/jessanrod3/vertebralcolumndataset/data
- Baixe `Dataset_spine.csv`

**Opção B - Kaggle API:**
```bash
pip install kaggle
kaggle datasets download -d jessanrod3/vertebralcolumndataset
unzip vertebralcolumndataset.zip -d data/
```

---

## 💻 Como Usar

### Executar o Notebook

```bash
jupyter notebook notebooks/Projeto_2_Modelagem_Coluna_Vertebral.ipynb
```

### Executar Células Sequencialmente

1. Abra o notebook no Jupyter
2. Execute cada célula com `Shift + Enter`
3. Ou execute todas: `Cell → Run All`

### Scripts Python (opcional)

```python
# Exemplo de uso dos módulos
from src.data_processing import load_and_clean_data
from src.models import train_logistic_regression
from src.evaluation import evaluate_model

# Carregar dados
X_train, X_test, y_train, y_test = load_and_clean_data('data/column_2C_weka.csv')

# Treinar modelo
model = train_logistic_regression(X_train, y_train)

# Avaliar
metrics = evaluate_model(model, X_test, y_test)
print(metrics)
```

---

## 📈 Resultados

### Performance dos Modelos

| Modelo | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|--------|----------|-----------|--------|----------|---------|
| Baseline | 67.7% | - | - | - | 0.500 |
| Naive Bayes | 83.9% | 0.855 | 0.952 | 0.901 | 0.892 |
| Naive Bayes (Otimizado) | 85.5% | 0.867 | 0.952 | 0.907 | 0.901 |
| Regressão Logística | 85.5% | 0.875 | 0.952 | 0.912 | 0.912 |
| **Regressão Logística (Otimizada)** | **87.1%** | **0.886** | **0.952** | **0.918** | **0.921** |

### 🏆 Modelo Campeão: Regressão Logística Otimizada
- **F1-Score**: 0.918
- **Melhoria sobre baseline**: +28.6%
- **Hiperparâmetros**: C=10, penalty='l2', solver='liblinear', class_weight='balanced'

### Features Mais Importantes

1. 🥇 `degree_spondylolisthesis` (coef: +1.45)
2. 🥈 `pelvic_incidence` (coef: +0.89)
3. 🥉 `sacral_slope` (coef: +0.67)

### Principais Insights

✅ **Dados limpos**: Sem valores ausentes ou duplicatas  
✅ **Separação clara**: Diferenças estatisticamente significativas entre classes (p < 0.05)  
✅ **Outliers mantidos**: Valores extremos são clinicamente relevantes  
✅ **Multicolinearidade moderada**: VIF < 10 para todas as features  
✅ **Balanceamento**: Dataset levemente desbalanceado (2.1:1), tratado com `class_weight='balanced'`  

---

## 🛠️ Tecnologias Utilizadas

### Linguagem
- Python 3.8 até 3.11

### Bibliotecas Principais

**Análise de Dados:**
- pandas 2.0+
- numpy 1.24+
- scipy 1.11+

**Visualização:**
- matplotlib 3.7+
- seaborn 0.12+

**Machine Learning:**
- scikit-learn 1.3+
- statsmodels 0.14+
- pycaret 3.0+

**Ambiente:**
- jupyter 1.0+
- notebook 7.0+

---

## 🔬 Metodologia

### 1. Análise Exploratória de Dados (EDA)

- ✅ Inspeção do esquema e tipos de dados
- ✅ Estatísticas descritivas
- ✅ Tratamento de valores ausentes e duplicatas
- ✅ Identificação e análise de outliers (IQR)
- ✅ Testes de normalidade (Shapiro-Wilk, KS, D'Agostino)
- ✅ Análise de correlações (Pearson)
- ✅ Visualizações (histogramas, boxplots, pairplots, heatmaps)
- ✅ Testes estatísticos (t-test, Mann-Whitney, Cohen's d)

### 2. Preparação dos Dados

- 📊 Divisão: 60% treino, 20% validação, 20% teste
- 🔄 Normalização: StandardScaler
- 🔍 Verificação de multicolinearidade (VIF)
- 🎯 Codificação da variável target (LabelEncoder)

### 3. Modelagem

**Baseline:**
- DummyClassifier (estratégia majoritária)

**Modelos Implementados:**
- Naive Bayes Gaussiano
- Regressão Logística

**Interpretação:**
- Coeficientes da Regressão Logística
- Importância de features

### 4. Avaliação

**Métricas:**
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC
- Matriz de Confusão
- Curvas ROC e Precision-Recall

**Diagnósticos:**
- Análise de resíduos
- Análise de erros (FP vs FN)
- Casos de incerteza

### 5. Otimização

**Técnicas Aplicadas:**
- ✅ Validação Cruzada (5-fold)
- ✅ Grid Search (Regressão Logística)
- ✅ Random Search (Naive Bayes)
- ✅ PyCaret (comparação automática de modelos)

**Hiperparâmetros Tunados:**
- `C` (regularização)
- `penalty` (L1/L2)
- `solver` (algoritmo de otimização)
- `class_weight` (balanceamento)
- `var_smoothing` (Naive Bayes)

---


## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

### Licença dos Dados

O dataset está licenciado sob **Database Contents License (DbCL) v1.0** e é de uso livre para fins acadêmicos, desde que devidamente citado.

**Citação:**
```bibtex
@misc{Dua:2019,
  author = "Dua, Dheeru and Graff, Casey",
  year = "2019",
  title = "{UCI} Machine Learning Repository",
  url = "http://archive.ics.uci.edu/ml",
  institution = "University of California, Irvine, School of Information and Computer Sciences"
}
```

---

## 📚 Referências

### Dataset
- Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. Irvine, CA: University of California, School of Information and Computer Science.
- Dr. Henrique da Mota - Centre Médico-Chirurgical de Réadaptation des Massues, Lyon, France

### Artigos Científicos
- Rocha-Neto, A. R., & Barreto, G. A. (2009). "On the Application of Ensembles of Classifiers to the Diagnosis of Pathologies of the Vertebral Column: A Comparative Analysis"

### Livros
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*. Springer.
- Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. O'Reilly Media.

### Documentação
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Statsmodels Documentation](https://www.statsmodels.org/)
- [PyCaret Documentation](https://pycaret.org/)

---

## 🙏 Agradecimentos

- UCI Machine Learning Repository pela disponibilização do dataset
- Dr. Henrique da Mota pela coleta e curadoria dos dados
- Comunidade open-source pelas excelentes bibliotecas Python

<div align="center">

**Desenvolvido por Andrey Garcia e Andrey de Matos**

</div>
