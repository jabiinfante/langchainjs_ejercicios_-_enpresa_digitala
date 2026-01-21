/**
 * =============================================================================
 * EJERCICIO: Poeta Cibernético con Tools y Structured Output
 * =============================================================================
 *
 * Este ejercicio combina dos conceptos importantes de LangChain:
 * - Tools (herramientas): Funciones que el LLM puede invocar para realizar tareas
 * - Structured Output: Forzar al LLM a responder con un formato JSON específico
 *
 * ENUNCIADO INICIAL (problema a resolver):
 * ─────────────────────────────────────────────────────────────────────────────
 * import { ChatMistralAI } from "@langchain/mistralai";
 *
 * const llm = new ChatMistralAI({
 *   model: "mistral-large-latest",
 *   temperature: 0.2,
 * });
 *
 * const prompt = `Eres un poeta cybernetico. Respondes en castellano.
 *   Tu estilo es oscuro y meláncolico.
 *   Necesito que además del poema me devuelvas 3 palabras clave del poema así
 *   como el número exacto de palabras del texto que generes.`;
 *
 * const response = await llm.invoke(prompt);
 * console.dir(response, { depth: null, colors: true });
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * PROBLEMA: El LLM no puede contar palabras con precisión (alucinará el número).
 * SOLUCIÓN: Usar una Tool que cuente las palabras de forma precisa.
 *
 * CONCEPTOS CLAVE:
 * - tool(): Función de LangChain para crear herramientas que el LLM puede usar
 * - bindTools(): Conectar herramientas a un modelo LLM
 * - tool_calls: El LLM indica qué herramienta quiere usar y con qué argumentos
 * - ToolMessage: Mensaje que contiene el resultado de ejecutar una herramienta
 * - withStructuredOutput(): Forzar formato de respuesta final
 *
 * FLUJO DEL EJERCICIO:
 * 1. Definir una herramienta para contar palabras
 * 2. El LLM genera un poema y solicita contar sus palabras (tool_call)
 * 3. Ejecutamos la herramienta y devolvemos el resultado (ToolMessage)
 * 4. El LLM genera la respuesta final estructurada con el conteo correcto
 * =============================================================================
 */

import {
  AIMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { ChatMistralAI } from "@langchain/mistralai";

// =============================================================================
// PASO 1: Definir la herramienta (Tool) para contar palabras
// =============================================================================
// Las Tools son funciones que el LLM puede decidir invocar.
// El LLM NO ejecuta la función directamente, solo indica que quiere usarla.
// Nosotros ejecutamos la función y le devolvemos el resultado.

const wordCountTool = tool(
  ({ texto }) => {
    // Esta función se ejecuta cuando procesamos el tool_call del LLM
    process.stdout.write("Contando palabras...");
    const words = texto.trim().split(/\s+/).filter(Boolean).length;
    process.stdout.write(`  [${words}]\n`);
    return words.toString();
  },
  {
    name: "contar_palabras",
    description:
      "Cuenta el numero de palabras en un texto, y devuelve solo el número.",
    // Zod define qué parámetros acepta la herramienta
    schema: z.object({ texto: z.string() }),
  },
);

// =============================================================================
// PASO 2: Configurar el modelo con las herramientas
// =============================================================================
// bindTools() conecta las herramientas al modelo.
// Ahora el LLM sabe que puede usar "contar_palabras" cuando lo necesite.

const llm = new ChatMistralAI({
  model: "mistral-large-latest",
  temperature: 0,
});

// IMPORTANTE: Pasar el array con las herramientas disponibles
const modelWithTools = llm.bindTools([wordCountTool]);

// =============================================================================
// PASO 3: Preparar la conversación inicial
// =============================================================================
// Usamos diferentes tipos de mensajes:
// - SystemMessage: Define el comportamiento/personalidad del LLM
// - HumanMessage: El mensaje del usuario
// - AIMessage: Respuesta del LLM (se añade después)
// - ToolMessage: Resultado de ejecutar una herramienta

const messages: Array<SystemMessage | AIMessage | HumanMessage | ToolMessage> =
  [
    new SystemMessage(
      "Eres un poeta cybernetico. Respondes en castellano. Tu estilo es oscuro y meláncolico.",
    ),
    new HumanMessage(
      `Necesito que generes un unico poema. Que uses las herramientas disponibles para contar las palabras de **ese** poema.
      Reutiliza siempre el primer poema que generes para hacer el conteo de palabras.`,
    ),
  ];

// =============================================================================
// PASO 4: Primera invocación - El LLM genera el poema y pide contar palabras
// =============================================================================
// El LLM responderá con:
// - content: El poema generado
// - tool_calls: Array indicando que quiere usar "contar_palabras"

const firstResponse = await modelWithTools.invoke(messages);

// Añadimos la respuesta del LLM al historial de mensajes
messages.push(firstResponse as AIMessage);

// =============================================================================
// PASO 5: Procesar los tool_calls (ejecutar las herramientas solicitadas)
// =============================================================================
// Si el LLM quiere usar herramientas, las ejecutamos y añadimos el resultado.

if (firstResponse.tool_calls && firstResponse.tool_calls.length > 0) {
  for (const call of firstResponse.tool_calls) {
    // Ejecutar la herramienta correspondiente
    let toolResult: string;
    if (call.name === "contar_palabras") {
      toolResult = await wordCountTool.invoke(call.args as { texto: string });
    } else {
      throw new Error(`Herramienta desconocida: ${call.name}`);
    }

    // Añadir el resultado como ToolMessage al historial
    // IMPORTANTE: tool_call_id debe coincidir con el id del tool_call
    messages.push(
      new ToolMessage({
        content:
          typeof toolResult === "string"
            ? toolResult
            : JSON.stringify(toolResult),
        tool_call_id: call.id!,
      }),
    );
  }

  // ===========================================================================
  // PASO 6: Generar la respuesta final estructurada
  // ===========================================================================
  // Ahora el LLM tiene el conteo real de palabras en el historial.
  // Usamos withStructuredOutput() para obtener un JSON con formato específico.

  const OutputSchema = z.object({
    poema: z
      .array(z.string())
      .describe(
        "El poema generado por el poeta cybernetico. Cada verso en una línea separada.",
      ),
    tematica: z.array(z.string()).length(3).describe("3 palabras clave del poema"),
    total_palabras: z.number().describe("El número exacto de palabras del poema"),
  });

  const modelWithOutput = llm.withStructuredOutput(OutputSchema);
  console.log("Generando salida final con conteo de palabras...");
  const final = await modelWithOutput.invoke(messages);

  console.dir(final, { depth: null, colors: true });
} else {
  // Si el LLM no pidió usar herramientas, mostramos su respuesta directa
  console.log(firstResponse.content);
}
