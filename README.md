# Modelización Matemática 2026 - Reto: Lo que arde

Repositorio del equipo para la resolución de los retos matemáticos y algorítmicos sobre prevención y actuación ante incendios forestales mediante teoría de grafos.

Sirve como solución al Concurso de Modelización Matemática (CMM-IMI). Edición 2026
El enunciado se especifica en el archivo `Lo_que_arde.pdf`. 

Más información sobre el concurso se puede encontrar aquí:
https://blogs.mat.ucm.es/cmm/edicion-2026/

## Archivos
`data`. Conjunto de ejemplos de uso proporcionados por la organización.

`core.py`. Establece la lectura y escritura de los grafos atendiendo a los datos proporcionados.

`engine.py`. Gestiona la lógica interna de los incendios y los bomberos.

`evolucion.py` Genera una imagen .png con la evolución del fuego en las t estapas.

`generar_datos_reto5.py` Genera los datos sintéticos (aristas, zonas y efectivos) a partir de los grafos del concurso.

`reto5_modelo.py` Define el bosque (Bosque), los parámetros globales (Parametros), la información disponible para los bomberos (Observacion) y la dinámica del fuego (Incendio).

`reto5_efectivos.py` Modela los efectivos (Efectivo) y su gestión durante la simulación (Brigada).

`reto5_politicas.py` Implementa el cálculo del riesgo y las políticas de asignación.




## Requisitos

Asegúrate de tener instalado Python 3.10 o superior. Instala las dependencias necesarias ejecutando:

```bash
pip install networkx matplotlib
