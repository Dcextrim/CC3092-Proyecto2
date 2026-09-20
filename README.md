# Proyecto 2: Detección de Lavado de Dinero en Secuencias de Remesas

Sistema de detección en dos etapas que aprende primero cómo se comporta un remitente normal y después qué distingue al lavado de dinero, sobre el conjunto [IBM Transactions for Anti Money Laundering](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml). Produce alertas con trazabilidad, es decir, para cada alerta el sistema señala qué transacciones concretas la activaron y redacta la razón en lenguaje natural.

> Para probar el MVP haz click [aquí](https://remittance-aml.onrender.com/) 

## El Problema

Los sistemas de detección basados en reglas que usan hoy los bancos y las empresas de remesas tienen tasas de falsos positivos del 95 al 99%. Un equipo de cumplimiento revisa cientos de alertas para encontrar un caso real, y los esquemas sofisticados pasan desapercibidos porque ninguna transacción aislada rompe una regla.

Lo que delata el lavado no es una transacción sino el patrón, la frecuencia que se dispara, los montos que se fraccionan justo debajo de un umbral de reporte, los destinos que cambian de golpe. Por eso el sistema juzga al remitente completo, mirando sus últimas 32 transacciones como una secuencia ordenada en el tiempo.

## El Sistema

### Etapa A: aprender qué es normal

Un autoencoder secuencial con atención se entrena únicamente con remitentes normales. Aprende a comprimir el comportamiento de un remitente en 128 números y a reconstruirlo desde ahí. Cuando recibe una secuencia que no se parece a nada de lo que vio, falla al reconstruirla, y ese fallo es la señal de anomalía.

### Etapa B: aprender qué es sospechoso

Un clasificador reutiliza el codificador que la etapa A entrenó sin etiquetas y le agrega una cabeza densa. El codificador se ajusta con un learning rate diez veces menor que el de la cabeza, para refinarlo sin que olvide lo que ya sabía.

### Fusión

Las dos señales se combinan con una regresión logística de dos entradas, que es auditable, se puede decir exactamente cuánto aportó cada etapa a cada decisión.

### Interpretabilidad

Los pesos de atención de la etapa B y el error de reconstrucción por paso de la etapa A indican qué transacciones pesaron más en la alerta.

## Estructura del Repositorio

```
config/                              Configuración y dependencias
├── model_config.yml                 Cada parámetro con la razón por la que se eligió
└── requirements.txt                 Dependencias de desarrollo

notebooks/
├── eda/01_datos_y_secuencias.ipynb                   De transacciones sueltas a secuencias
└── model_experiments/02_deteccion_dos_etapas.ipynb   Ambas etapas, ablación e interpretabilidad

src/
├── data_pipeline/
│   ├── construir.py                 Orquestador, del CSV crudo a los tensores
│   ├── sourcing/downloaders/        Descarga desde Kaggle, con backup
│   ├── sourcing/validators/         Comprobación de si PaySim admite secuencias
│   └── preprocessing/               Variables, secuencias y particiones
├── models/
│   ├── architectures/               Autoencoder secuencial y clasificador
│   ├── losses/                      Pérdida focal y entropía ponderada
│   └── metrics/                     Métricas apropiadas para desbalance extremo
├── training/
│   ├── scripts/                     Entrenamiento de cada etapa y experimento de ablación
│   └── utils/                       Semillas, lotes y submuestreo
├── inference/
│   ├── deployment/                  Aplicación Streamlit, exportación a ONNX y sus datos
│   └── utils/                       Vista legible de una secuencia y explicación escrita
└── utils/                           Rutas, configuración, credenciales, gráficos
                                     y empaquetado de artefactos

results/
├── models/                          Pesos entrenados y el modelo ONNX del MVP
├── plots/                           Figuras de ambos cuadernos
└── tables/                          Tabla de ablación, en detalle y resumida

docs/design_docs/reporte.pdf         Reporte ejecutivo
render.yaml                          Despliegue del MVP
```

> La carpeta `data/` se reconstruye ejecutando el primer cuaderno.

## Cómo Reproducirlo

### Requisitos

- Python 3.12
- [uv](https://docs.astral.sh/uv/) para gestionar el entorno
- **Una cuenta de Kaggle con llaves de API.**

### 1. Credenciales de Kaggle

Los dos conjuntos de datos se descargan desde Kaggle, y su API exige autenticación. En [kaggle.com](https://www.kaggle.com/settings) hay que entrar a la configuración de la cuenta, generar un token nuevo y guardarlo.

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

En Google Colab las mismas llaves se cargan como secretos del cuaderno, con los nombres `KAGGLE_USERNAME` y `KAGGLE_KEY`.

Si no encuentra credenciales, el código recurre a un espejo público de Hugging Face. Ese respaldo cubre **únicamente** el archivo de IBM AML, así que la sección del primer cuaderno que analiza PaySim se omite.

### 2. Crear el entorno

```bash
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install torch --index-url https://download.pytorch.org/whl/cpu
VIRTUAL_ENV=.venv uv pip install -r config/requirements.txt
```

La primera instalación de `torch` usa el índice de CPU a propósito. El paquete de PyPI incluye CUDA y pesa varias veces más, lo que no aporta nada en máquinas sin GPU de NVIDIA.

### 3. Registrar el kernel de Jupyter

```bash
VIRTUAL_ENV=.venv uv run python -m ipykernel install --user \
  --name remittance-aml --display-name "Remittance AML (py3.12)"
```

### 4. Ejecutar los cuadernos en orden

```bash
VIRTUAL_ENV=.venv uv run jupyter nbconvert --to notebook --execute --inplace \
  notebooks/eda/01_datos_y_secuencias.ipynb
VIRTUAL_ENV=.venv uv run jupyter nbconvert --to notebook --execute --inplace \
  notebooks/model_experiments/02_deteccion_dos_etapas.ipynb
```

El segundo cuaderno no depende del primero. Si no encuentra los tensores que el primero guarda, los reconstruye con las mismas funciones. El orden recomendado sigue siendo el numérico, porque el primero documenta las decisiones que el segundo da por hechas.

En Google Colab basta con abrir cada cuaderno y ejecutarlo. La primera celda clona el repositorio, instala lo que falte y configura las credenciales. **Con una GPU T4 gratuita el primer cuaderno tarda 40 segundos y el segundo menos de 15 minutos**.

Los cuadernos escriben sus resultados en el sistema de archivos de la máquina virtual, que se destruye al cerrar la sesión. La última celda de cada uno empaqueta el modelo, las figuras, las tablas y los datos del MVP en un zip y lo descarga. Ese archivo se descomprime sobre la copia local del repositorio.

### 5. Probar el MVP en local

```bash
VIRTUAL_ENV=.venv uv pip install -r src/inference/deployment/requirements.txt
VIRTUAL_ENV=.venv uv run streamlit run src/inference/deployment/app.py
```

La aplicación necesita `results/models/sistema.onnx` y los archivos de `src/inference/deployment/datos/`, que produce el segundo cuaderno.

## Resultados

Con un presupuesto de revisión del 1% del volumen, sobre 21,346 remitentes del conjunto de prueba con 292 casos de lavado. Tres semillas por configuración.

| Configuración                            | PR-AUC    | Casos detectados | Precisión |
| ---------------------------------------- | --------- | ---------------- | --------- |
| Etapa A sola, error de reconstrucción    | 0.040     | 8%               | 10%       |
| **Clasificador entrenado desde cero**    | **0.696** | **59%**          | **74%**   |
| Codificador congelado                    | 0.404     | 34%              | 50%       |
| Ajuste parcial desde la etapa A          | 0.604     | 51%              | 69%       |
| Ajuste parcial más fusión con la etapa A | 0.603     | 51%              | 68%       |
| Etapa A con learning rate uniforme       | 0.676     | 58%              | 73%       |

Frente a las tasas de falsos positivos del 95 al 99% que reportan los sistemas por reglas, que dos de cada tres alertas sean reales cambia la naturaleza del trabajo del equipo de cumplimiento.

La configuración de transferencia recomendada, reutilizar el codificador de la etapa A y ajustarlo con un learning rate reducido, pierde nueve puntos de PR-AUC frente a un clasificador que no usa la etapa A.

La última fila de la tabla permite decir de dónde vienen esos nueve puntos, porque comparte la inicialización con una configuración y el learning rate con la otra. **Solo dos puntos vienen de partir de la etapa A, y siete de entrenar el codificador diez veces más lento.** El error de diseño no estuvo en reutilizar la representación sino en frenarla, y la causa de fondo es que reconstruir comportamiento normal y clasificar lavado piden representaciones distintas.

Un experimento adicional midió si la transferencia rinde cuando hay pocas etiquetas, que es lo que la teoría predice. La curva resultó invertida, el perjuicio es máximo con 136 casos etiquetados y cae al crecer los datos.

El detalle completo está en [`docs/design_docs/reporte.pdf`](docs/design_docs/reporte.pdf) y en el segundo cuaderno. Para probar el sistema, accede al siguiente enlace: https://remittance-aml.onrender.com/

## Documentación adicional

- [`docs/design_docs/reporte.pdf`](docs/design_docs/reporte.pdf) es el reporte ejecutivo
- [`config/model_config.yml`](config/model_config.yml) reúne todos los parámetros con su justificación
