# Flujo de IA:

El flujo sería preprocesar datos, crear modelos optimizados, analizar métricas, predecir con los modelos generados, observar XAI, analizar drifting, generar datos sintéticos si es necesario, predecir sobre estos datos sintéticos, reentrenar los modelos con datos nuevos (o sintéticos en este caso) en el caso de ver drifting.

---

## Notas:
- **ABRIR CARPETA DE SCRIPTS**
- **NOTAS**: si existieran datos temporales, la columna debe llamarse "Timestamp" debido a funcionalidades internas.
- **NOTAS II**: se debe de conocer el nombre de la columna a predecir. Por defecto, los scripts suponen que se llama "Label".

---

## Descripción de Scripts

- **Preprocessor_exe.py**: 
    - Esta clase preprocesa todo el dataframe para asegurar que no haya valores faltantes. Utiliza un conjunto de datos de entrenamiento y devuelve un dataframe imputado.
    - Asume que los valores categóricos son booleanos numéricos.
    - Al llamar a la clase, se debe pasar el conjunto de datos de entrenamiento y el nombre de la columna objetivo. La variable donde se guarda el objeto será un imputador entrenado con TF.
    - Para procesar un nuevo dataframe con el objeto entrenado, se debe llamar a la función `Preprocessor.predictor(dataframe_to_fill)`.
    - Para instanciar el objeto, ha de pasarse al mismo el dataframe a preprocesar y la columna a predecir. Si se quiere guardar el dataframe se puede llamar a la función `Preprocessor`.

- **Preprocessor_caller.py**:
    - Esta clase llama a la clase explicada arriba. Es a través de esta clase que se ha de llamar al imputador ya que se encarga de tratar al dataframe que se introduce.
    - Para obtener el dataframe imputado se instanciará el objeto como el anterior: pasando el dataframe a procesar y la columna objetivo.

- **model_creator_exe.py**:
    - Creará tres modelos (XGB, Naïve Bayes, SVC) optimizados sobre un dataframe que se preprocesa gracias al preprocesador (`preprocessor_exe.py`) contenido.
    - El objeto `Model_Executer` necesita como parámetros de entrada el dataframe a preprocesar y el nombre de la columna a predecir ("Label" en el caso del dataset utilizado en el proyecto).
    - El objeto lanzará la imputación del dataframe, la normalización/preprocesamiento, y tendrá los siguientes atributos que deberían ser cargados a mlflow: los modelos (`objeto.booster`, `objeto.nb`, `objeto.svc`), y los parámetros de los modelos que están en un diccionario (`me.metrics_dict`). 
    - Así mismo, se pueden guardar los modelos en formato pickle (`.pkl`).

    Esta función recibe tres argumentos:
    - **model_to_save**: El modelo que se desea guardar.
    - **path**: La ruta donde se desea guardar el modelo.
    - **filename**: El nombre del archivo en el que se guardará el modelo.

    La función comienza obteniendo el directorio de trabajo actual utilizando `os.getcwd()`. Luego, combina la ruta proporcionada con el directorio de trabajo actual utilizando `os.path.join()` para asegurarse de que la ruta sea válida. Obtiene la fecha actual en el formato "dd_mm_aaaa" utilizando `datetime.datetime.now().strftime("%d_%m_%Y")`.

    Verifica si ya existe un directorio con el nombre de la fecha en la ruta proporcionada. Si existe, intenta guardar el modelo dentro de este directorio. Si no existe, crea un nuevo directorio con ese nombre y guarda el modelo utilizando `joblib.dump()` en un archivo `.pkl`.

- **final_data_generator.py**: 
    - Generador de datos artificiales. No preprocesa los datos del dataframe para poder generar los datos de la forma más fidedigna posible a los originales.
    - Los datos generados se les incluye una columna "Label" (objetivo de predicción) que se puede eliminar si el usuario lo desea.
    - Estos dataframes se guardan como `.csv`. Se puede acceder a los datos artificiales a través del atributo `objeto.new_data`. Si se quiere guardar el dataframe como `.csv`, se deberá llamar a la instancia `objeto.save_new_data(nuevos_datos)`.

    **Uso** (generar, evaluar y guardar datos):
    ```python
    sd = Synthetic_Data(df) 
    new_df = sd.new_data
    cols = new_df.columns
    sd.evaluation(*cols)
    sd.save_new_data(new_df, "name_of_the_folder")
    ```

### Notas sobre final_data_generator:
El equipo de orquestación y API debería pasar estos datos nuevos a Evidently para ver si hay drifting, conectar también los datos nuevos a los modelos de ML para ver cómo predicen. Se podría unir estos datos a los originales y ver cómo funcionan en producción.

- **evidently_drifting.py**:
    - El script utiliza la biblioteca Evidently para monitorear el desplazamiento de datos, que puede afectar el rendimiento del modelo con el tiempo.
    - Está diseñado para ser ejecutado desde la línea de comandos con argumentos que especifican las ubicaciones de los datos de referencia y actuales, y una lista de nombres de columnas a analizar.
    - El script imprime información en la consola para depuración y maneja excepciones. En caso de fallo, cambia a una configuración predeterminada.
    - Está pensado para operaciones MLOps donde el monitoreo del desplazamiento de datos es crucial para garantizar el rendimiento de los modelos implementados.

### Notas sobre evidently_drifting:
La idea es mantener un dataset estático como referencia y con el orquestador lanzar la creación de datos sintéticos, quitar la columna "Label", analizar el drifting y luego predecir si los datos tienen buena calidad. Si se tienen datos reales, se recomienda tener un set de referencia y alimentar los datos de producción a este script para analizar el drifting.

- **_optimizer_exe.py**:
    - Son optimizadores para cada modelo ejecutado y creado. No se utilizan individualmente, sino que `model_creator` los llama y optimiza a partir de ellos.

- **XAI_exe.py**:
    - Objeto capaz de explicar por qué un modelo XGB predice de la forma que lo hace. Utiliza modelados SHAP.

- **Retrainer_exe.py**:
    - Una vez pasado el modelo, los datos nuevos serán el conjunto de reentreno de los modelos. Este reentreno será gestionado por el usuario en base a lo que se observe en Evidently. Si se percibe un drifting, este módulo será ejecutado para que los modelos superen el problema de drifting.
