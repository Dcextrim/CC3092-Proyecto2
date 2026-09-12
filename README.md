# Proyecto 2 - Deteccion de fraude con PaySim

Proyecto para CC3092 - Deep Learning y Sistemas Inteligentes. El sistema representa historiales temporales por cuenta y compara un autoencoder de normalidad con un clasificador supervisado que reutiliza su encoder.

## Estructura

- `notebooks/`: notebook principal del proyecto.
- `artifacts/`: pesos, scores de prueba y pesos de atencion generados por la ultima ejecucion.
- `data/`: directorio local opcional para PaySim. Esta ignorado por Git.

## Ejecutar

1. Cree y active un entorno virtual.
2. Instale dependencias: `python -m pip install -r requirements.txt`.
3. Coloque `PS_20174392719_1491204439457_log.csv` dentro de `data/`, o deje el CSV dos niveles arriba del repositorio como en este espacio de trabajo.
4. Abra `notebooks/Proyecto2_AML_PaySim.ipynb` y ejecute todas las celdas.

En Google Colab, suba el CSV o su ZIP a `/content` antes de ejecutar el notebook. Se recomienda una GPU T4 para entrenar; la prueba local se hizo en CPU.
https://www.kaggle.com/datasets/ealaxi/paysim1
https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

## Resultado de la prueba local

La ejecucion completa produjo los artefactos actuales. En esta primera configuracion, el clasificador supervisado desde cero supero al modelo de dos etapas: PR-AUC 0.9604 frente a 0.9497 en la fusion. Este resultado se conserva para la ablacion requerida; no se afirma que el preentrenamiento aporte valor hasta mejorar y volver a evaluar la Etapa A.

PaySim es un dataset sintetico de fraude financiero. Los resultados no equivalen a una validacion de lavado de dinero real ni de operaciones de remesas guatemaltecas.
