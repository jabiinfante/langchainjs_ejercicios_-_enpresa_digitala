/**
 * =============================================================================
 * EJERCICIO: Clasificador de Comentarios con Structured Output
 * =============================================================================
 *
 * Este ejercicio demuestra cómo usar LangChain para clasificar comentarios
 * de usuarios utilizando "Structured Output" (salida estructurada).
 *
 * CONCEPTOS CLAVE:
 * - Zod: Librería para definir y validar esquemas de datos en TypeScript
 * - withStructuredOutput(): Método que fuerza al LLM a responder con un JSON
 *   que cumple exactamente con el esquema definido
 * - batch(): Procesar múltiples inputs en paralelo de forma eficiente
 *
 * FLUJO DEL EJERCICIO:
 * 1. Definir un esquema con Zod que describe la estructura de respuesta esperada
 * 2. Configurar el modelo LLM con withStructuredOutput()
 * 3. Enviar comentarios al modelo para que los analice y clasifique
 * 4. Recibir respuestas estructuradas (JSON válido según el esquema)
 * =============================================================================
 */

import { z } from "zod";
import { ChatMistralAI } from "@langchain/mistralai";
import { initChatModel } from "langchain";
import { comments } from "./helpers/comments-mock";

// =============================================================================
// PASO 1: Definir el esquema de respuesta con Zod
// =============================================================================
// Zod nos permite definir la estructura exacta que esperamos del LLM.
// Esto garantiza que la respuesta sea un JSON válido y tipado.

const CommentSchema = z.object({
  // Nivel de lenguaje ofensivo (0 = ninguno, 5 = muy ofensivo)
  profanity_level: z.number().min(0).max(5),

  // Sentimiento general del comentario
  sentiment: z.enum(["positive", "neutral", "negative"]),

  // Temas identificados en el comentario (mínimo 1)
  topics: z.array(z.string()).min(1)
}).describe("Esquema para analizar comentarios de usuarios");

// TypeScript infiere automáticamente el tipo desde el esquema Zod
type Comment = z.infer<typeof CommentSchema>;

// =============================================================================
// PASO 2: Ejemplo de validación con Zod (sin LLM)
// =============================================================================
// Podemos usar el esquema para validar datos manualmente.
// Si los datos no cumplen el esquema, Zod lanzará un error.

const result = CommentSchema.parse({
  profanity_level: 5,
  sentiment: "neutral",
  topics: ["technology", "education"]
});

// =============================================================================
// PASO 3: Configurar el modelo LLM con Structured Output
// =============================================================================
// withStructuredOutput() recibe el esquema Zod y configura el modelo para
// que SIEMPRE responda con un JSON que cumpla ese esquema.

// Opción A: Usar Mistral AI directamente
const llm = new ChatMistralAI({
  model: "mistral-large-latest",
  streaming: true,
  maxTokens: 1000,
}).withStructuredOutput(CommentSchema);

// Opción B: Usar initChatModel() para inicializar cualquier modelo de forma genérica
// Esto permite cambiar fácilmente entre proveedores (OpenAI, Google, Anthropic, etc.)
const llm2 = (await initChatModel('gemini-2.5-flash', {
  modelProvider: 'google-genai'
})).withStructuredOutput(CommentSchema);

// =============================================================================
// PASO 4: Procesar comentarios en lote (batch)
// =============================================================================
// batch() permite enviar múltiples inputs al modelo de forma eficiente.
// maxConcurrency limita cuántas peticiones se hacen en paralelo.

const result2 = await llm.batch(
  comments.map(c => c.content),  // Extraer solo el contenido de cada comentario
  { maxConcurrency: 2 }          // Máximo 2 peticiones simultáneas
);

// =============================================================================
// PASO 5: Mostrar resultados
// =============================================================================
// Cada resultado es un objeto tipado que cumple con CommentSchema

result2.forEach(async (result, index) => {
  console.log(`Respuesta a la pregunta ${comments[index].uuid} (${comments[index].content}):`);
  console.dir(result, { depth: null, colors: true });
});