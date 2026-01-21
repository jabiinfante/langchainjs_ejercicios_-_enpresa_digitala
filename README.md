# Ejercicios del Curso de LangChain

Este repositorio contiene los ejercicios prácticos del curso de **LangChain con TypeScript**.

## Recursos del Curso

- **Slides (demo)**: [https://jabiinfante.github.io/slides-langchain.js](https://jabiinfante.github.io/slides-langchain.js)
  - Código fuente: [https://github.com/jabiinfante/slides-langchain.js](https://github.com/jabiinfante/slides-langchain.js)
- **Cliente del Agente (demo)**: [https://jabiinfante.github.io/langchain.js-agent-client-dummy/](https://jabiinfante.github.io/langchain.js-agent-client-dummy/)
  - Código fuente: [https://github.com/jabiinfante/langchain.js-agent-client-dummy](https://github.com/jabiinfante/langchain.js-agent-client-dummy)

## Requisitos Previos

### 1. Cuentas y API Keys necesarias

#### Mistral AI (Requerido)

1. Crear cuenta en [https://console.mistral.ai/](https://console.mistral.ai/)
2. Ir a "API Keys" y generar una nueva key
3. Guardar la key como `MISTRAL_API_KEY`

#### LangSmith (Requerido para trazabilidad)

1. Crear cuenta en [https://smith.langchain.com/](https://smith.langchain.com/)
2. Ir a "Settings" → "API Keys" y crear una nueva key
3. Guardar la key como `LANGCHAIN_API_KEY`

#### Google AI / Gemini (Opcional)

1. Ir a [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Crear una API Key
3. Guardar la key como `GOOGLE_API_KEY`

#### Qdrant (Opcional - solo para ejercicios 04 y 05)

1. Crear cuenta en [https://cloud.qdrant.io/](https://cloud.qdrant.io/)
2. Crear un cluster gratuito
3. Obtener la URL del cluster y la API Key
4. Guardar como `QDRANT_URL` y `QDRANT_API_KEY`

### 2. Configurar variables de entorno

Crear un archivo `.env` en la raíz del proyecto con el siguiente contenido:

```env
# === Mistral AI (Requerido) ===
MISTRAL_API_KEY=tu_api_key_de_mistral

# === LangSmith (Requerido para trazabilidad) ===
LANGCHAIN_API_KEY=tu_api_key_de_langsmith
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=curso-langchain

# === Google AI / Gemini (Opcional) ===
GOOGLE_API_KEY=tu_api_key_de_google

# === Qdrant (Opcional - solo para ejercicios 04 y 05) ===
QDRANT_URL=https://tu-cluster.qdrant.io
QDRANT_API_KEY=tu_api_key_de_qdrant
```

### 3. Instalar dependencias

```bash
npm install
```

## Catálogo de Ejercicios

| #   | Script                | Archivo                     | Descripción                                                                                                                                                              |
| --- | --------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 01  | `npm run 01:comments` | `01_comments_classifier.ts` | **Clasificador de Comentarios** - Analiza comentarios usando Structured Output con Zod. Demuestra `withStructuredOutput()` y `batch()` para procesar múltiples inputs.   |
| 02  | `npm run 02:poet`     | `02_cyber_poet.ts`          | **Poeta Cibernético** - Combina Tools y Structured Output. El LLM genera un poema y usa una herramienta para contar palabras con precisión.                              |
| 03  | `npm run 03:homework` | `03_homework_maker.ts`      | **Generador de Tareas** - Agente multi-tool con bucle agentic manual. Usa Wikipedia, Calculator y WordCount para generar tareas escolares adaptadas al nivel del alumno. |
| 04  | `npm run 04:agent`    | `04_webserver_for_agent.ts` | **Servidor Web con Agente** - Integra un agente ReAct con Fastify y SSE para streaming en tiempo real. Incluye memoria persistente y herramientas RAG.                   |
| 05  | `npm run 05:indexer`  | `05_mdn-vector-indexer.ts`  | **Indexador de Documentación** - Pipeline de indexación RAG que carga documentación de MDN, la divide en chunks y la almacena en Qdrant.                                 |

## Estructura del Proyecto

```
src/
├── 01_comments_classifier.ts  # Structured Output + batch
├── 02_cyber_poet.ts           # Tools + Structured Output
├── 03_homework_maker.ts       # Agentic Loop + PromptTemplate + múltiples tools
├── 04_webserver_for_agent.ts  # Fastify + SSE + Agent con memoria
├── 05_mdn-vector-indexer.ts   # Web scraping + chunking + Qdrant
│
├── agents_wrapper/
│   └── agent.ts               # Wrapper del agente con streaming y contextSchema
│
└── helpers/
    ├── comments-mock.ts       # Datos de prueba para ejercicio 01
    ├── helper.ts              # Utilidad promptUser() para input interactivo
    ├── middlewares.ts         # Middleware trimMessages para limitar contexto
    └── tools.ts               # Herramientas: wordCount, wikipedia, exchangeRates, storageKnowledge
```

## Conceptos por Ejercicio

### 01 - Clasificador de Comentarios

- **Zod**: Definición de esquemas para validación
- **withStructuredOutput()**: Forzar respuestas JSON estructuradas
- **batch()**: Procesar múltiples inputs en paralelo

### 02 - Poeta Cibernético

- **tool()**: Crear herramientas personalizadas
- **bindTools()**: Conectar herramientas al modelo
- **tool_calls**: El LLM solicita usar herramientas
- **ToolMessage**: Devolver resultados de herramientas

### 03 - Generador de Tareas

- **Agentic Loop**: Bucle while que procesa tool_calls hasta completar
- **PromptTemplate**: Plantillas con variables dinámicas
- **Múltiples Tools**: Wikipedia, Calculator, WordCount
- **promptUser()**: Input interactivo en consola

### 04 - Servidor Web con Agente

- **createAgent()**: Crear agente ReAct con herramientas
- **Checkpointer (SQLite)**: Memoria persistente de conversaciones
- **SSE (Server-Sent Events)**: Streaming de respuestas al cliente
- **contextSchema**: Inyección de dependencias a las tools
- **dynamicSystemPromptMiddleware**: Prompts que cambian en cada invocación

### 05 - Indexador de Documentación

- **CheerioWebBaseLoader**: Web scraping de HTML
- **RecursiveCharacterTextSplitter**: División de documentos en chunks
- **Embeddings**: Representación vectorial de texto
- **QdrantVectorStore**: Base de datos vectorial para búsqueda semántica
- **Deduplicación**: Eliminar vectores existentes antes de re-indexar

## Orden Recomendado

1. **01_comments_classifier** - Conceptos básicos de Structured Output
2. **02_cyber_poet** - Introducción a Tools
3. **03_homework_maker** - Agentic Loop manual con múltiples tools
4. **04_webserver_for_agent** - Agente completo con servidor web y streaming
5. **05_mdn-vector-indexer** - Indexación de documentación para RAG (opcional, mejora el 04)
