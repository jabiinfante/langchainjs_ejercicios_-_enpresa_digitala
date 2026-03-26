/**
 * =============================================================================
 * EJERCICIO: Generador de Tareas Escolares con Agente Multi-Tool
 * =============================================================================
 *
 * Este ejercicio demuestra un caso de uso avanzado combinando múltiples conceptos:
 * - Bucle de agente manual (agentic loop): El modelo decide cuándo usar herramientas
 * - PromptTemplate: Plantillas de prompts con variables dinámicas
 * - Múltiples herramientas: Wikipedia, Calculator, Word Counter
 * - Structured Output: Respuesta final en formato JSON estructurado
 * - Input interactivo: Configuración mediante prompts de usuario
 *
 * CASO DE USO:
 * Un estudiante necesita ayuda con sus deberes. El agente:
 * 1. Recibe la configuración (nivel, asignatura, idioma, extensión)
 * 2. Busca información relevante en Wikipedia
 * 3. Genera el texto adaptado al nivel del alumno
 * 4. Verifica que cumple con la extensión requerida
 * 5. Devuelve un informe estructurado
 *
 * CONCEPTOS CLAVE:
 * - PromptTemplate.fromTemplate(): Crear prompts con variables {variable}
 * - bindTools(): Conectar múltiples herramientas al modelo
 * - Agentic Loop: Bucle while que procesa tool_calls hasta completar la tarea
 * - withStructuredOutput(): Generar respuesta final estructurada
 *
 * FLUJO DEL EJERCICIO:
 * 1. Recoger configuración del usuario (nivel, asignatura, idioma, etc.)
 * 2. Formatear el system prompt con las variables
 * 3. Ejecutar bucle de agente:
 *    - Invocar modelo
 *    - Si hay tool_calls: ejecutar herramientas y añadir resultados
 *    - Repetir hasta que no haya más tool_calls o se alcance el límite
 * 4. Generar informe final estructurado
 * =============================================================================
 */

import {
  AIMessage,
  BaseMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";
import { ChatMistralAI } from "@langchain/mistralai";
import { Calculator } from "@langchain/community/tools/calculator";
import { WikipediaQueryRun } from "@langchain/community/tools/wikipedia_query_run";
import { PromptTemplate } from "@langchain/core/prompts";
import { z } from "zod";
import { promptUser } from "./helpers/helper";
import { wordCountTool } from "./helpers/tools";

// =============================================================================
// PASO 1: Recoger configuración del usuario
// =============================================================================
// promptUser() es un helper que muestra un prompt en consola y recoge la respuesta.
// El segundo parámetro es el valor por defecto si el usuario presiona Enter.

const config = {
  level: await promptUser("¿En qué curso está el alumno?", "Primero de la ESO"),
  subject: await promptUser("¿Cuál es la asignatura de la tarea?", "Historia"),
  language: await promptUser("¿En qué idioma está la tarea?", "Euskera"),
  number_words: await promptUser(
    "¿Cuántas palabras debe tener la tarea?",
    "500",
  ),
  question: await promptUser(
    "Introduce la pregunta que quieres que responda el alumno:",
    "Pequeña redacción con la historia del Imperio Romano",
  ),
};

// =============================================================================
// PASO 2: Configurar el modelo y las herramientas
// =============================================================================

const model = new ChatMistralAI({
  model: "mistral-large-latest",
  temperature: 0.2, // Baja temperatura para respuestas más consistentes
});

// Wikipedia: Buscar información de referencia
const wikipediaTool = new WikipediaQueryRun({
  topKResults: 3,           // Máximo 3 resultados
  maxDocContentLength: 1500, // Limitar contenido para no exceder contexto
});

// Calculator: Para cálculos matemáticos si la tarea lo requiere
const calculatorTool = new Calculator();

// Conectar todas las herramientas al modelo
const modelWithTools = model.bindTools([
  wordCountTool,    // Contar palabras (verificar extensión)
  wikipediaTool,    // Buscar información
  calculatorTool,   // Cálculos matemáticos
]);

// =============================================================================
// PASO 3: Crear el prompt template con variables
// =============================================================================
// PromptTemplate permite crear prompts reutilizables con placeholders {variable}
// que se sustituyen al llamar a format()

const systemPromptTemplate = PromptTemplate.fromTemplate(`
Eres un asistente educativo experto que ayuda a estudiantes a completar sus tareas escolares.
Tu objetivo es generar contenido educativo de alta calidad adaptado al nivel del alumno.

## CONTEXTO DEL ALUMNO
- **Nivel educativo:** {level}
- **Asignatura:** {subject}
- **Idioma de la tarea:** {language}
- **Extensión requerida:** {number_words} palabras (margen: ±10%)

## HERRAMIENTAS DISPONIBLES

1. **wikipedia-api** - Búsqueda de información
   - USAR SIEMPRE para obtener datos precisos y verificables
   - Buscar en el idioma apropiado para obtener mejores resultados
   - Consultar múltiples términos si es necesario para cubrir el tema

2. **calculator** - Calculadora matemática
   - USAR para cualquier cálculo numérico (fechas, porcentajes, estadísticas)

3. **contar_palabras** - Verificación de extensión
   - USAR OBLIGATORIAMENTE antes de dar la respuesta final
   - Si el conteo está fuera del rango permitido, ajustar el texto

## FLUJO DE TRABAJO

1. **INVESTIGAR**: Busca información en Wikipedia sobre el tema solicitado
2. **PLANIFICAR**: Organiza las ideas principales según el nivel educativo
3. **REDACTAR**: Escribe el texto en {language}, adaptando vocabulario y complejidad
4. **VERIFICAR**: Cuenta las palabras y ajusta si es necesario
5. **ENTREGAR**: Proporciona el texto final verificado

## REGLAS DE CALIDAD

- **Adaptación al nivel**: Un alumno de primaria necesita lenguaje simple; uno de bachillerato puede manejar conceptos más complejos
- **Estructura clara**: Usa párrafos bien organizados con introducción, desarrollo y conclusión
- **Precisión**: Todos los datos deben provenir de Wikipedia, no inventes información
- **Originalidad**: Redacta con tus propias palabras, no copies textualmente de Wikipedia
- **Idioma**: TODO el contenido debe estar en {language}, incluyendo términos técnicos cuando sea posible

## IMPORTANTE
- NO entregues la tarea sin verificar el conteo de palabras
- Si el texto es muy corto, amplía con más detalles o ejemplos
- Si el texto es muy largo, sintetiza manteniendo la información esencial
`);

// =============================================================================
// PASO 4: Preparar mensajes iniciales
// =============================================================================

const { question, ...params } = config;

// Formatear el prompt sustituyendo las variables
const systemPrompt = await systemPromptTemplate.format(params);

const messages: BaseMessage[] = [
  new SystemMessage(systemPrompt),
  new HumanMessage(question),
];

// =============================================================================
// PASO 5: Bucle de agente (Agentic Loop)
// =============================================================================
// Este bucle implementa el patrón ReAct manualmente:
// - El modelo genera una respuesta (puede incluir tool_calls)
// - Si hay tool_calls, ejecutamos las herramientas y añadimos los resultados
// - Repetimos hasta que el modelo no pida más herramientas

let iteracion = 0;
const MAX_ITERACIONES = 6; // Límite de seguridad para evitar bucles infinitos

while (iteracion < MAX_ITERACIONES) {
  iteracion++;
  console.log(`\n🔄 Iteración ${iteracion}...`);

  // Invocar el modelo con el historial de mensajes
  const response = await modelWithTools.invoke(messages);
  messages.push(response);

  // Si no hay tool_calls, el modelo ha terminado
  if (!response.tool_calls || response.tool_calls.length === 0) {
    console.log(`\n✅ Tarea completada`);
    break;
  }

  // Procesar cada tool_call solicitado por el modelo
  for (const call of response.tool_calls) {
    console.log(`\n🔧 Tool: ${call.name}`);
    console.log(`   📝 Args: ${JSON.stringify(call.args)}`);

    let resultado: string;

    // Ejecutar la herramienta correspondiente
    if (call.name === "contar_palabras") {
      resultado = `${await wordCountTool.invoke(call.args as { texto: string })}`;
    } else if (call.name === "wikipedia-api") {
      console.log(`   🌐 Buscando en Wikipedia: "${call.args.input}"`);
      const wikiResults = await wikipediaTool.invoke(call.args.input as string);
      resultado =
        typeof wikiResults === "string"
          ? wikiResults
          : JSON.stringify(wikiResults);
    } else if (call.name === "calculator") {
      console.log(`   🧮 Calculando: "${call.args.input}"`);
      resultado = `${await calculatorTool.invoke(call.args.input as string)}`;
    } else {
      resultado = `Error: herramienta "${call.name}" no reconocida`;
    }

    // Añadir el resultado de la herramienta como ToolMessage
    messages.push(
      new ToolMessage({
        content: resultado,
        tool_call_id: call.id!,
      }),
    );
  }
}

// =============================================================================
// PASO 6: Generar informe final estructurado
// =============================================================================
// Una vez completada la tarea, pedimos un informe en formato JSON.

const homeworkReportSchema = z.object({
  texto: z.string().describe("El texto completo de la tarea para el alumno"),
  numero_palabras: z.number().describe("Número total de palabras en el texto"),
  nivel: z.string().describe("Nivel educativo del alumno"),
  urls_wikipedia: z
    .array(z.string())
    .describe("URLs de Wikipedia consultadas (si las hay)"),
});

console.log(`\n📝 Generando informe final...`);
messages.push(new HumanMessage("Genera el informe final con la tarea completada para el alumno."));

const modelWithOutput = model.withStructuredOutput(homeworkReportSchema);
const informe = await modelWithOutput.invoke(messages);

console.log("\n" + "=".repeat(60));
console.log("INFORME FINAL");
console.log("=".repeat(60));
console.dir(informe, { depth: null, colors: true });
