# Informe tecnico - Sistema RAG sobre la Ley 769 de 2002

## 1. Datos generales

- Estudiante: Juan David Tejedor Medina
- Fecha: 21 de junio de 2026
- Repositorio: https://github.com/juandavidtejedormedina-hub/Proyecto-RAG
- Documento base: `LEY 769 DE 2002.pdf`
- Nombre del asistente: LexTransito

## 2. Objetivo del proyecto

El objetivo del proyecto es construir un asistente conversacional real basado
en RAG, capaz de responder preguntas en lenguaje natural sobre un documento
legal especializado. El sistema debe recuperar fragmentos relevantes del PDF,
generar una respuesta fundamentada y citar las paginas consultadas.

## 3. Documento seleccionado

Se selecciono la Ley 769 de 2002, Codigo Nacional de Transito Terrestre de
Colombia. El documento tiene 123 paginas y corresponde a una fuente legal real
relacionada con transito, movilidad, licencias, comparendos, motociclistas,
peatones y reglas de circulacion.

La seleccion es adecuada porque:

- Es un documento legal especializado.
- Supera el requisito minimo de 30 paginas.
- Permite preguntas practicas de usuarios no tecnicos.
- Requiere respuestas precisas y trazables por pagina.

## 4. Usuario objetivo y caso de uso

El usuario objetivo es un ciudadano, conductor, instructor, estudiante de
conduccion o persona interesada en consultar normas generales de transito en
Colombia.

Caso de uso principal: el usuario pregunta en lenguaje natural sobre reglas de
transito y LexTransito responde con base en los fragmentos recuperados del PDF,
sin inventar informacion y citando paginas al final.

## 5. Arquitectura del sistema

El sistema esta implementado en `RAG.PY` y usa una arquitectura RAG local:

1. Carga del PDF con PyMuPDF.
2. Extraccion del texto pagina por pagina.
3. Division del texto en chunks con solapamiento.
4. Generacion de embeddings locales con Ollama y el modelo `all-minilm`.
5. Almacenamiento del indice vectorial local en `vectorstore/`.
6. Recuperacion de los fragmentos mas relevantes para cada pregunta.
7. Generacion de respuesta con Ollama y el modelo `llama3.2:3b`.
8. Cita final de paginas consultadas.

La base vectorial se genera localmente y no se sube a GitHub porque puede
reconstruirse ejecutando el sistema.

## 6. Componentes principales

### 6.1 Carga del documento

El PDF se carga desde:

```text
LEY 769 DE 2002.pdf
```

La libreria PyMuPDF permite abrir el documento y extraer texto pagina por
pagina, lo cual permite conservar trazabilidad entre cada fragmento y su pagina
de origen.

### 6.2 Creacion de chunks

El texto de cada pagina se divide en fragmentos de tamano controlado. El sistema
usa:

```text
RAG_CHUNK_SIZE=1400
RAG_CHUNK_OVERLAP=250
```

Esto reduce el riesgo de cortar ideas importantes y permite recuperar contexto
relevante para cada pregunta.

### 6.3 Embeddings locales

El sistema usa Ollama para generar embeddings locales con:

```text
OLLAMA_EMBEDDING_MODEL=all-minilm
```

Esto permite cumplir la etapa de generacion de embeddings sin depender de una
API externa ni de una clave de pago.

### 6.4 Base vectorial local

Los chunks, paginas y embeddings se almacenan en:

```text
vectorstore/ley_769_index.json
```

La carpeta `vectorstore/` esta ignorada por Git mediante `.gitignore`.

### 6.5 Recuperacion de contexto

El sistema recupera los fragmentos mas relevantes segun la similitud entre la
pregunta y los embeddings almacenados. Adicionalmente, combina senales lexicas
para mejorar la precision en preguntas legales en espanol.

### 6.6 Generacion de respuesta

El modelo local de generacion es:

```text
OLLAMA_MODEL=llama3.2:3b
```

El prompt del sistema define el rol de LexTransito y obliga a responder solo
con informacion del contexto recuperado.

## 7. Prompt del asistente

LexTransito tiene las siguientes reglas:

- Responder solo con informacion del contexto recuperado.
- Decir explicitamente cuando la respuesta no este en el contexto.
- No inventar sanciones, excepciones ni paginas.
- No dar asesoria legal personalizada.
- Citar al final las paginas consultadas.

## 8. Ejecucion del sistema

Instalar dependencias:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Descargar modelos locales en Ollama:

```powershell
ollama pull llama3.2:3b
ollama pull all-minilm
```

Construir el indice vectorial:

```powershell
python RAG.PY --build-index --retriever ollama
```

Ejecutar el asistente:

```powershell
python RAG.PY --generation-mode ollama
```

Para terminar la sesion:

```text
salir
```

## 9. Preguntas de prueba

### Pregunta 1

Pregunta:

```text
Que regula la Ley 769 de 2002?
```

Respuesta de referencia:

```text
La Ley 769 de 2002 regula la circulacion de peatones, usuarios, pasajeros,
conductores, motociclistas, ciclistas, agentes de transito y vehiculos por las
vias publicas o privadas abiertas al publico, asi como la actuacion y
procedimientos de las autoridades de transito.

Paginas consultadas: p. 2.
```

### Pregunta 2

Pregunta:

```text
Los menores pueden viajar en el asiento delantero?
```

Respuesta de referencia:

```text
Los menores de diez anos no pueden viajar en el asiento delantero. Los menores
de dos anos deben viajar en el asiento posterior usando una silla que garantice
su seguridad.

Paginas consultadas: p. 58.
```

### Pregunta 3

Pregunta:

```text
Cuando se puede suspender o cancelar una licencia por embriaguez?
```

Respuesta de referencia:

```text
La licencia puede suspenderse cuando el conductor se encuentre en estado de
embriaguez o bajo el efecto de drogas alucinogenas determinado por la autoridad
competente. Tambien puede cancelarse por reincidencia.

Paginas consultadas: p. 31, p. 32.
```

### Pregunta 4

Pregunta:

```text
Que obligaciones tienen los motociclistas segun el documento?
```

Respuesta de referencia:

```text
Los motociclistas deben transitar ocupando un carril, usar luces direccionales,
mantener luces delanteras y traseras encendidas, y conductor y acompanante
deben usar casco.

Paginas consultadas: p. 63, p. 64.
```

### Pregunta 5

Pregunta:

```text
Las camaras pueden servir como prueba de una infraccion?
```

Respuesta de referencia:

```text
Las camaras de video y equipos electronicos de lectura que permitan identificar
con precision el vehiculo o conductor son validos como prueba de una infraccion
y pueden dar lugar a un comparendo.

Paginas consultadas: p. 86.
```

## 10. Evidencia de funcionamiento

Durante las pruebas locales, el sistema:

- Cargo el PDF correctamente.
- Creo 304 chunks.
- Genero embeddings locales con Ollama.
- Guardo el indice en `vectorstore/ley_769_index.json`.
- Respondio preguntas con paginas citadas.

Ejemplo de salida:

```text
Sistema RAG listo.
Asistente: LexTransito
Documento: LEY 769 DE 2002.pdf
Chunks indexados: 304
Recuperacion: ollama_embeddings
Generacion: ollama
Ollama: disponible | modelo: llama3.2:3b
```

## 11. Limitaciones

- La calidad de la respuesta depende de la extraccion de texto del PDF.
- Si el PDF tiene saltos o texto mal reconocido, algunos chunks pueden quedar
  incompletos.
- El modelo local `llama3.2:3b` es liviano; puede ser menos preciso que modelos
  grandes en linea.
- La respuesta debe ser revisada si se usara como orientacion legal formal.

## 12. Conclusiones

El proyecto cumple con las actividades de un sistema RAG: carga de documento,
chunking, embeddings, almacenamiento vectorial, recuperacion de contexto y
respuesta conversacional. La solucion funciona de forma local con Ollama, por
lo que no depende de una API key externa. Ademas, el asistente LexTransito
incluye reglas de prompt para evitar invenciones y citar paginas consultadas.

