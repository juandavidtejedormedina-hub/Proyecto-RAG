# Sistema RAG sobre la Ley 769 de 2002

Repositorio: https://github.com/juandavidtejedormedina-hub/Proyecto-RAG

Informe tecnico: [INFORME_TECNICO.md](INFORME_TECNICO.md)

Estado de entrega: codigo fuente y README listos. Video pendiente.

## Nombre del estudiante y fecha

- Estudiante: Juan David Tejedor Medina
- Fecha: 21 de junio de 2026

## Documento seleccionado y justificacion

Documento: `LEY 769 DE 2002.pdf`

Este proyecto usa la Ley 769 de 2002, Codigo Nacional de Transito Terrestre
de Colombia. Es un documento legal real, especializado y con 123 paginas, por
lo que cumple el requisito de tener mas de 30 paginas de contenido tecnico o
legal. El tema es adecuado para un sistema RAG porque las respuestas deben ser
precisas, trazables y soportadas en articulos del documento.

## Persona usuaria objetivo y caso de uso

La persona usuaria objetivo es un ciudadano, conductor, estudiante de conduccion
o instructor que necesita consultar dudas generales sobre normas de transito en
lenguaje natural.

Caso de uso: una persona pregunta sobre cinturones de seguridad, motocicletas,
licencias, comparendos o reglas de circulacion, y el asistente responde con
base en el contexto recuperado del PDF, citando las paginas consultadas.

## Nombre y rol del asistente

El asistente se llama **LexTransito**.

Rol: asistente legal especializado en la Ley 769 de 2002, Codigo Nacional de
Transito Terrestre de Colombia.

Reglas del prompt:

- Responde unicamente con informacion del contexto recuperado.
- Si la respuesta no esta en el documento, lo dice explicitamente.
- No inventa informacion.
- Cita al final de cada respuesta las paginas consultadas.

## Arquitectura del sistema

El sistema implementa las actividades pedidas:

1. Carga el documento PDF con PyMuPDF.
2. Divide el contenido en chunks por pagina.
3. Genera embeddings locales con Ollama usando `all-minilm`.
4. Guarda los chunks y embeddings en una base vectorial local en `vectorstore/`.
5. Ejecuta un bucle conversacional por consola para responder preguntas.
6. Usa IA local con Ollama, modelo `llama3.2:3b`, para redactar respuestas.
   Si Ollama no esta disponible, puede usar respuesta extractiva local como respaldo.

La base vectorial se guarda localmente en formato JSON para facilitar la entrega
y evitar dependencias pesadas. La carpeta `vectorstore/` no se sube al repo
porque se puede reconstruir ejecutando el sistema.

## Instalacion

Desde la carpeta del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Configuracion local sin API key

Este proyecto puede ejecutarse sin API key usando recuperacion local y Ollama.

1. Instala Ollama desde https://ollama.com/download
2. Abre PowerShell y descarga un modelo pequeno:

```powershell
ollama pull llama3.2:3b
ollama pull all-minilm
```

3. Copia `.env.example` como `.env`.
4. Deja la configuracion local:

```env
RAG_RETRIEVER=ollama
RAG_GENERATION_MODE=auto
OLLAMA_MODEL=llama3.2:3b
OLLAMA_EMBEDDING_MODEL=all-minilm
OLLAMA_URL=http://localhost:11434
```
## Orden de ejecucion

Construir el indice vectorial:

```powershell
python RAG.PY --build-index --retriever ollama
```

Ejecutar el chat:

```powershell
python RAG.PY
```

Tambien puedes reconstruir el indice y abrir el chat en un solo paso:

```powershell
python RAG.PY --rebuild
```

Para forzar respuesta con Ollama:

```powershell
python RAG.PY --generation-mode ollama
```

Para forzar respuesta extractiva sin ningun modelo generativo:

```powershell
python RAG.PY --generation-mode extractive
```

Para salir del chat escribe:

```text
salir
```

## Cinco preguntas de prueba y respuestas esperadas

### 1. Que regula la Ley 769 de 2002?

Respuesta esperada:
La Ley 769 de 2002 regula la circulacion de peatones, usuarios, pasajeros,
conductores, motociclistas, ciclistas, agentes de transito y vehiculos en las
vias publicas o privadas abiertas al publico. Tambien regula la actuacion y
procedimientos de las autoridades de transito.

Paginas consultadas: p. 2.

### 2. Los menores pueden viajar en el asiento delantero?

Respuesta esperada:
El documento indica que los menores de diez anos no pueden viajar en el asiento
delantero. Tambien senala que los menores de dos anos solo pueden viajar en el
asiento posterior usando una silla que garantice su seguridad, cuando viajen
unicamente con el conductor.

Paginas consultadas: p. 58.

### 3. Cuando se puede suspender o cancelar una licencia por embriaguez?

Respuesta esperada:
La licencia de conduccion se puede suspender cuando el conductor se encuentre
en estado de embriaguez o bajo el efecto de drogas alucinogenas determinado por
la autoridad competente. La licencia se puede cancelar por reincidencia al
conducir en cualquier grado de embriaguez o bajo el efecto de drogas
alucinogenas.

Paginas consultadas: p. 31, p. 32.

### 4. Que obligaciones tienen los motociclistas segun el documento?

Respuesta esperada:
El documento indica que las motocicletas deben transitar ocupando un carril,
usar luces direccionales, mantener luces delanteras y traseras encendidas en
todo momento en vias de uso publico, y que conductor y acompanante deben usar
casco. El acompanante tambien debe usar la prenda reflectiva exigida.

Paginas consultadas: p. 63, p. 64.

### 5. Las camaras pueden servir como prueba de una infraccion?

Respuesta esperada:
Si. El documento senala que las ayudas tecnologicas como camaras de video y
equipos electronicos de lectura que permitan identificar con precision el
vehiculo o el conductor son validas como prueba de ocurrencia de una infraccion
de transito y pueden dar lugar a la imposicion de un comparendo.

Paginas consultadas: p. 86.

Link del video: 
