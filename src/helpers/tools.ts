/**
 * =============================================================================
 * HERRAMIENTAS (TOOLS) PARA AGENTES DE LANGCHAIN
 * =============================================================================
 *
 * Este archivo contiene las herramientas que los agentes pueden usar para
 * realizar tareas específicas. Las Tools permiten al LLM interactuar con
 * el mundo exterior: APIs, bases de datos, cálculos, etc.
 *
 * CONCEPTOS CLAVE:
 * - tool(): Función para crear herramientas personalizadas
 * - description: El LLM usa esta descripción para decidir CUÁNDO usar la tool
 * - schema: Define los parámetros que acepta la herramienta (validados con Zod)
 * - config: Segundo parámetro de la tool que permite acceder al contexto runtime
 *
 * BUENAS PRÁCTICAS PARA DESCRIPTIONS:
 * - Ser específico sobre cuándo usar la herramienta
 * - Indicar qué tipo de información devuelve
 * - Usar mayúsculas para enfatizar casos de uso obligatorios
 *
 * RUNTIME CONTEXT:
 * Las tools pueden acceder al contexto del agente a través del parámetro `config`.
 * Esto permite inyectar dependencias (como vectorStore, DB connections, etc.)
 * en tiempo de ejecución, evitando estado global y haciendo las tools más testables.
 * =============================================================================
 */
import { CohereRerank } from "@langchain/cohere";

import { WikipediaQueryRun } from "@langchain/community/tools/wikipedia_query_run";
import { DocumentInterface } from "@langchain/core/documents";
import { VectorStore } from "@langchain/core/vectorstores";
import { ChatMistralAI } from "@langchain/mistralai";
import { HumanMessage, SystemMessage, tool } from "langchain";
import { z } from "zod";

// =============================================================================
// CONTEXT SCHEMA: Define la estructura del contexto del agente
// =============================================================================
// Este esquema se usa en createAgent() para tipar el contexto que se pasa
// al invocar el agente. Las tools acceden a él vía config.context
//
// Ejemplo de uso:
//   const agent = createAgent({ ..., contextSchema: agentContextSchema });
//   agent.invoke({ messages }, { context: { vectorStore: myVectorStore } });

export const agentContextSchema = z.object({
  vectorStore: z
    .custom<VectorStore>()
    .describe("Vector store para búsquedas RAG"),
});

// Tipo TypeScript inferido del esquema
export type AgentContext = z.infer<typeof agentContextSchema>;

// =============================================================================
// TOOL: Contador de Palabras
// =============================================================================
// Herramienta síncrona simple para contar palabras en un texto.
// Útil porque los LLMs no pueden contar con precisión.

export const wordCountTool = tool(
  ({ texto }) => {
    process.stdout.write("Contando palabras...");
    const words = texto.trim().split(/\s+/).filter(Boolean).length;
    process.stdout.write(`  [${words}]\n`);
    return words.toString();
  },
  {
    name: "contar_palabras",
    description:
      "Cuenta el número exacto de palabras en un texto. USAR SIEMPRE que necesites saber la cantidad de palabras de un texto, ya que no puedes contarlas con precisión por ti mismo.",
    schema: z.object({
      texto: z.string().describe("El texto del cual contar las palabras"),
    }),
  },
);

// =============================================================================
// TOOL: Wikipedia
// =============================================================================
// Herramienta pre-construida de LangChain para buscar en Wikipedia.
// Útil para obtener información general y actualizada sobre cualquier tema.

export const wikipediaTool = new WikipediaQueryRun({
  topKResults: 3, // Número máximo de resultados a devolver
  maxDocContentLength: 4000, // Longitud máxima del contenido por documento
});

// =============================================================================
// TOOL: Tasas de Cambio Actuales
// =============================================================================
// Consulta la API de Frankfurter para obtener tasas de cambio en tiempo real.
// API gratuita y sin autenticación: https://www.frankfurter.app/

export const getExchangeRatesTool = tool(
  async ({ base, symbols }) => {
    const url = `https://api.frankfurter.app/latest?from=${base}&to=${symbols.join(",")}`;
    const response = await fetch(url);
    const data = await response.json();
    return JSON.stringify(data);
  },
  {
    name: "get_exchange_rates",
    description:
      "Obtiene las tasas de cambio ACTUALES desde una moneda base a otras monedas. Usar cuando el usuario pregunte por el valor actual de una divisa o quiera convertir cantidades entre monedas HOY.",
    schema: z.object({
      base: z
        .string()
        .describe("Código ISO de la moneda base (ej: EUR, USD, GBP, JPY)"),
      symbols: z
        .array(z.string())
        .describe(
          "Array de códigos ISO de las monedas destino (ej: ['USD', 'GBP'])",
        ),
    }),
  },
);

// =============================================================================
// TOOL: Tasas de Cambio Históricas
// =============================================================================
// Consulta tasas de cambio de una fecha específica en el pasado.
// Útil para comparar evolución de divisas o consultas sobre fechas concretas.

export const getHistoricalRatesTool = tool(
  async ({ date, base, symbols }) => {
    const url = `https://api.frankfurter.dev/v1/${date}?from=${base}&to=${symbols.join(",")}`;
    console.log(`Consultando tasas de cambio históricas con URL: ${url}`);
    const response = await fetch(url);
    const data = await response.json();
    console.log(data);
    return JSON.stringify(data);
  },
  {
    name: "get_historical_rates",
    description:
      "Obtiene tasas de cambio de una fecha PASADA específica. Usar cuando el usuario pregunte por el valor de una divisa en una fecha concreta, o quiera comparar la evolución de una moneda entre dos fechas.",
    schema: z.object({
      date: z.string().describe("Fecha en formato YYYY-MM-DD (ej: 2024-01-15)"),
      base: z.string().describe("Código ISO de la moneda base (ej: EUR, USD)"),
      symbols: z
        .array(z.string())
        .describe("Array de códigos ISO de las monedas destino"),
    }),
  },
);

// =============================================================================
// TOOL: Base de Conocimiento sobre Web Storage
// =============================================================================
// Herramienta RAG que consulta una base de datos vectorial con documentación
// sobre sistemas de almacenamiento en navegadores (localStorage, IndexedDB, etc.)
//
// IMPORTANTE: Esta tool accede al vectorStore desde el contexto del agente
// en lugar de usar una variable global. Esto hace la tool más testable y
// permite usar diferentes vector stores según el contexto.
//
// El vectorStore se pasa al invocar el agente:
//   agent.invoke({ messages }, { context: { vectorStore: myVectorStore } });

const mistralMini = new ChatMistralAI({
  model: "mistral-tiny",
}).withStructuredOutput(
  z.object({
    querys: z
      .array(z.string())
      .describe("Consulta del usuario para buscar en la base de conocimiento"),
  }),
);

export const storageKnowledgeTool = tool(
  async ({ query }, config) => {
    
    process.stdout.write(`Buscando en base de conocimiento: "${query}"\n`);

    const querys = await mistralMini.invoke([
      new SystemMessage(
        `A partir del mensaje del usuario, necesito una colección de palabras (una o dos) relevantes semanticamente relacionadas, para encontrar lo que necesita el usuario, buscando en una bbdd vectorial con guiones de películas.`,
      ),
      new HumanMessage(query),
    ]);

    process.stdout.write(
      `  → Busquedas generadas por Mistral: ${JSON.stringify(querys)}\n`,
    );

    // Acceder al vectorStore desde el contexto del agente
    // config.context contiene los valores pasados en { context: {...} } al invocar
    const vectorStore = (config as any).context?.vectorStore as VectorStore;

    if (!vectorStore) {
      return "Error: No se ha configurado el vector store en el contexto del agente.";
    }
    const docs: Array<DocumentInterface> = [];

    for (const q of querys.querys) {
      process.stdout.write(`  → Buscando en vector store con query: "${q}"\n`);
      const retrievedDocs = await vectorStore.similaritySearch(q, 3);
      process.stdout.write(
        `    → ${retrievedDocs.length} documentos encontrados\n`,
      );

      if (retrievedDocs.length > 0) {
        docs.push(...retrievedDocs);
      }
    }

    if (docs.length === 0) {
      return `No se encontró información relevante sobre "${query}".`;
    }

    const reranker = new CohereRerank({
      apiKey: process.env.COHERE_API_KEY,
      model: "rerank-v3.5",
      topN: 5,
    });

    const rerankedDocs = await reranker.compressDocuments(docs, query);
    console.log(`  → Documentos reordenados por relevancia con Cohere Rerank`);

    // Formatear los documentos recuperados como contexto
    const context = rerankedDocs
      .map((doc) => doc.pageContent)
      .join("\n\n---\n\n");

    return `Información encontrada sobre "${query}":\n\n${context}`;
  },
  {
    name: "storage_knowledge",
    description:
      "Consulta la base de conocimiento sobre peliculas de Christopher Nolan. USAR OBLIGATORIAMENTE cuando el usuario pregunte sobre: Interstellar, Inception. Devuelve información relevante extraída de documentos relacionados con esas películas.",
    schema: z.object({
      query: z
        .string()
        .describe(
          "Pregunta o tema a buscar sobre peliculas de Christopher Nolan",
        ),
    }),
  },
);
