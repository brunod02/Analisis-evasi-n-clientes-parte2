## Propósito del Análisis
El objetivo de este análisis fue predecir el abandono de clientes (`Churn`) utilizando un conjunto de datos que contiene diversas características demográficas, de servicio telefónico, de internet y de cuenta.

## Estructura del Proyecto

La estructura recomendada para este proyecto es la siguiente:

```
proyecto_churn/
├── data/
│   ├── datos_tratados.csv
├── notebooks/
│   └── Analisis_Churn.ipynb
├── models/
│   └── # Aquí se podrían guardar modelos entrenados (ej. .pkl)
├── README.md
└── requirements.txt # Archivo para listar dependencias
```

## Preparación de Datos

### Clasificación de Variables
Las variables del conjunto de datos se clasificaron inicialmente en dos tipos:
- **Variables Categóricas:** Aquellas que representan categorías discretas, como `customer.gender`, `customer.Partner`, `internet.InternetService`, `account.PaymentMethod`, entre otras. Estas fueron listadas en la variable `variables_categoricas` en el cuaderno.
- **Variables Numéricas:** Aquellas que contienen valores continuos o discretos con un orden significativo, como `customer.tenure`, `account.Charges.Monthly`, y `account.Charges.Total`. Estas fueron listadas en la variable `variables_numericas`.

### Codificación y Normalización
Para preparar los datos para el modelado:
- **`OneHotEncoder`:** Se utilizó para transformar las variables categóricas explicativas (presentes en `X`) en un formato numérico binario, creando nuevas columnas por cada categoría ùnica. Esto evita que el modelo asuma una relación ordinal inexistente entre las categorías.
- **`LabelEncoder`:** Se aplicó a la variable objetivo `Churn` (`y`) para convertir sus valores categóricos ('Yes'/'No') a numéricos (1/0), lo que es un requisito para la mayoría de los algoritmos de aprendizaje supervisado.
- **`MinMaxScaler`:** Para el modelo K-Nearest Neighbors (KNN), fue crucial normalizar las características numéricas del conjunto de entrenamiento (`X_train`) y de prueba (`X_test`). Esto asegura que todas las características contribuyan equitativamente a la distancia calculada por el algoritmo, evitando que las características con rangos de valores más grandes dominen el cálculo de la distancia.

### División de Datos
El conjunto de datos (`X`, `y`) se dividió en conjuntos de entrenamiento y prueba utilizando `sklearn.model_selection.train_test_split` con las siguientes consideraciones:
- **`X_train`, `X_test`, `y_train`, `y_test`:** Los datos se separaron en 75% para entrenamiento y 25% para prueba.
- **`stratify=y`:** Se aplicó estratificación sobre la variable objetivo `y` para asegurar que la proporción de clientes que abandonan (Churn) y no abandonan sea la misma en los conjuntos de entrenamiento y prueba que en el conjunto de datos original, lo cual es fundamental en problemas con clases desbalanceadas.
- **`random_state=42`:** Se fijó un estado aleatorio para asegurar la reproducibilidad de la división de los datos.

## Análisis Exploratorio de Datos (EDA) e Insights

Durante el proceso de EDA, se realizaron diversas visualizaciones para comprender la distribución de los datos y las relaciones entre las variables, especialmente en relación con la variable objetivo `Churn`:

- **Matriz de Correlación:** Se generó un mapa de calor (`sns.heatmap`) para visualizar la correlación entre todas las variables, incluyendo la variable `Churn` (celda `vlc6hMDyjN2A`). Esto permitió identificar rápidamente las variables más y menos correlacionadas con el abandono de clientes.
- **Boxplots de Tiempo de Contrato vs. Cancelación:** Se utilizaron diagramas de caja (`sns.boxplot`) para comparar la distribución del tiempo de contrato (`customer.tenure`) y los cargos totales (`account.Charges.Total`) entre los clientes que cancelan y los que no. Estos gráficos mostraron que los clientes con menor tiempo de contrato y menores cargos totales son más propensos a cancelar.
- **Scatter Plot de Gasto Total vs. Cancelación:** Un diagrama de dispersión (`sns.scatterplot`) visualizó la relación entre `customer.tenure` y `account.Charges.Total`, coloreando los puntos segùn el estado de `Churn`. Este gráfico reforzó la idea de que los clientes de corta duración y bajo gasto son un segmento vulnerable.

### Insights Clave del EDA:
Los principales factores identificados que influyen en la probabilidad de abandono, basados en la correlación y el análisis exploratorio, incluyen:

1.  **Tipo de Contrato (Month-to-month contract):** Alta correlación positiva. Los clientes con contratos mensuales tienen una propensión significativamente mayor a cancelar.
2.  **Tiempo de Tenencia (`customer.tenure`):** Fuerte correlación negativa. Los clientes con menor tenencia son más propensos a cancelar.
3.  **Servicio de Internet (Fiber optic):** Correlación positiva. Los clientes con fibra óptica muestran una mayor propensión a la cancelación.
4.  **Soporte Técnico (`internet.TechSupport_No`):** Correlación positiva. La ausencia de soporte técnico se asocia con una mayor tasa de cancelación.
5.  **Seguridad Online (`internet.OnlineSecurity_No`):** Correlación positiva. Los clientes sin servicios de seguridad online son más propensos a cancelar.
6.  **Método de Pago (Electronic check):** Correlación positiva. Los clientes que pagan mediante cheque electrónico tienen una mayor probabilidad de cancelación.
7.  **Cargos Totales (`account.Charges.Total`):** Correlación negativa. Cargos totales más bajos se asocian con una mayor propensión a la cancelación.

## Justificación de Decisiones de Modelización

Se entrenaron y evaluaron cuatro modelos de clasificación:
*   **Dummy Classifier:** Como línea base, para establecer un umbral de rendimiento mínimo. Sufrió de *underfitting*, prediciendo solo la clase mayoritaria.
*   **Árbol de Decisión (`DecisionTreeClassifier`):** Con una profundidad máxima de 5 (`max_depth=5`) para controlar la complejidad. Este modelo mostró el mejor equilibrio entre métricas, con una Accuracy de 0.7882, Precision de 0.6063, Recall de 0.5739, F1-score de 0.5897 y un AUC de 0.8296. Fue el más adecuado para identificar clientes propensos a abandonar.
*   **Random Forest (`RandomForestClassifier`):** Con 300 estimadores (`n_estimators=300`) y `max_depth=5`. Alcanzó el AUC más alto (0.8402), pero su Recall para la clase Churn (0.4390) fue el más bajo entre los modelos predictivos, lo que significa que perdió un nùmero significativo de clientes que realmente abandonaban.
*   **K-Nearest Neighbors (`KNeighborsClassifier`):** Usando datos normalizados y con `n_neighbors=30` (optimizado mediante `GridSearchCV`). Fue competitivo pero ligeramente inferior al Árbol de Decisión y Random Forest en la mayoría de las métricas (Accuracy de 0.7814, Precision de 0.5976, Recall de 0.5375, F1-score de 0.5660 y un AUC de 0.8199).

### Conclusión y Recomendación

Considerando la necesidad de un equilibrio entre la identificación de clientes que abandonan y la precisión de esas predicciones, el **Árbol de Decisión con `max_depth=5` es el modelo recomendado.** Ofrece un buen balance y un rendimiento general robusto, además de ser más interpretable que el Random Forest. Si el objetivo principal fuera maximizar la precisión, el Random Forest podría ser una opción, pero su bajo recall para la clase positiva lo hace menos deseable para la detección temprana de abandono.

## Análisis de la Importancia de las Variables

El análisis de la importancia de las características de los modelos de Árbol de Decisión y Random Forest reveló los siguientes factores clave que influyen significativamente en la probabilidad de que un cliente abandone el servicio. Se ha tomado en cuenta la relevancia de las variables en ambos modelos para una visión más consolidada.

### Variables más influyentes (considerando Árbol de Decisión y Random Forest):

1.  **Tipo de Contrato (onehotencoder__account.Contract_Month-to-month):** Consistente como la variable más importante en el Árbol de Decisión (51.30%) y muy alta en Random Forest (16.66%). Indica que los clientes con contratos mensuales tienen una propensión significativamente mayor a cancelar.
2.  **Tiempo de Tenencia (remainder__customer.tenure):** Segunda variable más importante en el Árbol de Decisión (17.90%) y también muy alta en Random Forest (14.88%). Los clientes con menor tenencia son más propensos a cancelar.
3.  **Servicio de Internet Fibra Óptica (onehotencoder__internet.InternetService_Fiber optic):** Tercera variable más importante en el Árbol de Decisión (14.07%) y también alta en Random Forest (6.39%). Los clientes con fibra óptica muestran una mayor propensión a la cancelación.
4.  **Cargos Totales (remainder__account.Charges.Total):** Importante en el Árbol de Decisión (4.18%) y muy relevante en Random Forest (8.98%). Aunque no siempre es una causa directa, se correlaciona con la propensión al Churn, especialmente si los cargos son bajos (asociado a menor tenencia).
5.  **Falta de Soporte Técnico (onehotencoder__internet.TechSupport_No):** Relevante en el Árbol de Decisión (2.78%) y alta en Random Forest (7.88%). La ausencia de este servicio se asocia con una mayor tasa de cancelación.
6.  **Método de Pago Cheque Electrónico (onehotencoder__account.PaymentMethod_Electronic check):** Importante en el Árbol de Decisión (2.75%) y alta en Random Forest (4.85%). Los clientes que pagan mediante cheque electrónico tienen una mayor probabilidad de cancelación.
7.  **Falta de Seguridad Online (onehotencoder__internet.OnlineSecurity_No):** Importante en Random Forest (6.64%) y con cierta presencia en Árbol de Decisión (0.53%). Los clientes sin servicios de seguridad online son más propensos a cancelar.

Estos hallazgos son consistentes con los insights obtenidos en el Análisis Exploratorio de Datos y proporcionan una base sólida para el desarrollo de estrategias de retención dirigidas.

## Estrategias de Retención de Clientes Basadas en Factores Clave

A continuación, se presentan estrategias de retención de clientes directamente derivadas de los factores de cancelación identificados, buscando mitigar su impacto y mejorar la fidelización.

### Factores de Cancelación y Estrategias Propuestas:

1.  **Tipo de Contrato: Mes a Mes**
    *   **Impacto:** Clientes con contratos mensuales tienen una mayor propensión a la cancelación, probablemente debido a la falta de compromiso a largo plazo y la facilidad para cambiar de proveedor.
    *   **Estrategia de Retención:** Ofrecer incentivos atractivos para la renovación de contratos a plazos más largos (uno o dos años), como descuentos significativos en la tarifa mensual, mejoras gratuitas de servicio (por ejemplo, mayor velocidad de internet, canales premium) o dispositivos gratuitos/con descuento al firmar un contrato a término. Crear paquetes de fidelización que recompensen la permanencia.

2.  **Tiempo de Tenencia**
    *   **Impacto:** Los clientes con menor tiempo de tenencia son más propensos a cancelar, indicando que el período inicial es crítico para la retención.
    *   **Estrategia de Retención:** Implementar un programa de bienvenida robusto que incluya seguimiento proactivo (llamadas, emails) durante los primeros 3-6 meses para asegurar la satisfacción, resolver problemas rápidamente y educar sobre el uso óptimo de los servicios. Ofrecer un punto de contacto dedicado para nuevos clientes.

3.  **Servicio de Internet: Fibra Óptica**
    *   **Impacto:** Aunque la fibra óptica es un servicio avanzado, los clientes que lo tienen muestran una mayor tasa de cancelación. Esto podría deberse a expectativas no cumplidas, problemas de estabilidad o altos costos.
    *   **Estrategia de Retención:** Realizar encuestas de satisfacción específicas para usuarios de fibra óptica. Ofrecer soporte técnico prioritario y especializado para este segmento. Comunicar claramente los beneficios y la diferencia de la fibra óptica y justificar su valor. Monitorear proactivamente la calidad del servicio de fibra y actuar ante cualquier anomalía.

4.  **Falta de Soporte Técnico**
    *   **Impacto:** Los clientes sin soporte técnico son más propensos a cancelar, ya que no tienen un recurso para resolver problemas.
    *   **Estrategia de Retención:** Promocionar activamente los servicios de soporte técnico existentes, asegurando que los clientes estén conscientes de cómo acceder a ellos. Ofrecer paquetes de servicio que incluyan soporte técnico premium o asistencia remota. Invertir en la mejora de la calidad y tiempos de respuesta del soporte para reducir la frustración del cliente.

5.  **Falta de Seguridad Online**
    *   **Impacto:** Clientes sin servicios de seguridad online son más vulnerables y pueden sentir menos confianza en su proveedor, lo que lleva a la cancelación.
    *   **Estrategia de Retención:** Incluir servicios básicos de seguridad online (antivirus, firewall) de forma gratuita o a bajo costo como parte de los paquetes de internet. Educar a los clientes sobre los riesgos de seguridad y cómo la empresa puede protegerlos. Ofrecer actualizaciones a planes de seguridad más completos con incentivos.

6.  **Método de Pago: Cheque Electrónico**
    *   **Impacto:** Los clientes que pagan con cheque electrónico tienen una mayor tasa de cancelación. Esto podría indicar una menor estabilidad financiera o un proceso de pago menos automatizado que puede generar fricción.
    *   **Estrategia de Retención:** Incentivar el cambio a métodos de pago más convenientes y automatizados como tarjetas de crédito/débito o domiciliación bancaria, ofreciendo pequeños descuentos o recompensas por la configuración de pagos automáticos. Simplificar el proceso de pago electrónico para reducir la fricción.

7.  **Cargos Totales**
    *   **Impacto:** Un alto monto de cargos totales (acumulados a lo largo del tiempo) se correlaciona con la cancelación, lo que sugiere insatisfacción con el costo acumulado o percepción de poco valor por el dinero.
    *   **Estrategia de Retención:** Ofrecer revisiones periódicas de las cuentas de los clientes de alto valor, buscando optimizar sus planes de servicio para asegurar que estén obteniendo el mejor valor. Proponer descuentos por fidelidad a largo plazo o paquetes personalizados que se ajusten a sus necesidades y presupuesto, mitigando la percepción de altos costos acumulados.

## Instrucciones para Ejecutar el Cuaderno

Para ejecutar el cuaderno `Analisis_Churn.ipynb`, asegùrate de tener las siguientes bibliotecas instaladas y de colocar el archivo de datos correctamente.

### Bibliotecas Requeridas

Las bibliotecas principales utilizadas en el cuaderno son:
- `pandas`
- `numpy`
- `matplotlib.pyplot`
- `seaborn`
- `plotly.express`
- `sklearn.preprocessing` (OneHotEncoder, LabelEncoder, MinMaxScaler)
- `sklearn.compose` (make_column_transformer)
- `sklearn.model_selection` (train_test_split, GridSearchCV)
- `sklearn.dummy` (DummyClassifier)
- `sklearn.metrics` (accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_score, recall_score, f1_score, roc_auc_score)
- `sklearn.tree` (DecisionTreeClassifier, plot_tree)
- `sklearn.ensemble` (RandomForestClassifier)
- `sklearn.neighbors` (KNeighborsClassifier)
- `sklearn.inspection` (permutation_importance)

### Carga de Datos

El cuaderno espera encontrar el archivo de datos `datos_tratados.csv` en la carpeta `/content/` del entorno de ejecución (por ejemplo, Google Colab). Si estás ejecutando el cuaderno localmente o en un entorno diferente, asegùrate de:

1.  **Colocar el archivo `datos_tratados.csv` en la ruta `/content/`** o en la ubicación especificada por el código.
2. 
    ```
    datos = pd.read_csv('/content/datos_tratados.csv')
    ```
    Asegùrate de que la ruta (`'/content/datos_tratados.csv'`) refleje la ubicación real de tu archivo.
