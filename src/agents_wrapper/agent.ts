/**
 * =============================================================================
 * EJERCICIO: Wrapper de Agente LangChain con Streaming
 * =============================================================================
 *
 * Este archivo implementa un wrapper sobre el agente de LangChain que permite:
 * - Gestionar múltiples conversaciones simultáneas (por thread_id/uuid)
 * - Streaming de mensajes a clientes conectados vía SSE
 * - Memoria persistente de conversaciones (checkpointer)
 * - Integración con herramientas (tools) personalizadas
 * - Inyección de dependencias vía contextSchema (ej: vectorStore)
 *
 * ENUNCIADO INICIAL (versión simplificada sin LangChain):
 * ─────────────────────────────────────────────────────────────────────────────
 * import { AIMessage, BaseMessage, HumanMessage } from "@langchain/core/messages";
 *
 * export class Agent {
 *   // Handlers organizados por uuid (thread_id) para soportar múltiples conversaciones
 *   private messagesHandlers: Record<string, Array<(message: BaseMessage) => void>> = {};
 *
 *   messageReceived(message: string, clientConfig: { thread_id: string }) {
 *     const { thread_id } = clientConfig;
 *     const m = new HumanMessage(message);
 *     this.messagesHandlers[thread_id]?.forEach((handler) => handler(m));
 *
 *     // Simular respuesta del agente después de 6 segundos
 *     setTimeout(() => {
 *       const m = new AIMessage(`Respuesta del agente a "${message}"`);
 *       this.messagesHandlers[thread_id]?.forEach((handler) => handler(m));
 *     }, 6000);
 *   }
 *
 *   registerNewMessageHandler(handler: (message: BaseMessage) => void, uuid: string) {
 *     this.messagesHandlers[uuid] = this.messagesHandlers[uuid] || [];
 *     this.messagesHandlers[uuid].push(handler);
 *   }
 *
 *   unregisterNewMessageHandler(uuid: string, handler: (message: BaseMessage) => void) {
 *     this.messagesHandlers[uuid] = this.messagesHandlers[uuid].filter((h) => h !== handler);
 *   }
 * }
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * CONCEPTOS CLAVE:
 * - createAgent(): Crea un agente ReAct con herramientas y middleware
 * - Checkpointer: Guarda el estado de la conversación (memoria persistente)
 * - stream(): Permite recibir respuestas del agente de forma incremental
 * - dynamicSystemPromptMiddleware: Permite generar prompts dinámicos en cada invocación
 * - contextSchema: Define el contexto que se pasa a las tools (ej: vectorStore)
 * - messagesHandlers: Patrón pub/sub para notificar a clientes conectados
 *
 * FLUJO:
 * 1. Cliente se conecta y registra un handler (registerNewMessageHandler)
 * 2. Cliente envía mensaje (messageReceived)
 * 3. Agente procesa el mensaje y genera respuesta (puede usar tools)
 * 4. Cada chunk de respuesta se envía a los handlers registrados
 * 5. Cliente se desconecta (unregisterNewMessageHandler)
 * =============================================================================
 */

import { Calculator } from "@langchain/community/tools/calculator";
import { BaseMessage, HumanMessage } from "@langchain/core/messages";
import { VectorStore } from "@langchain/core/vectorstores";
import { BaseCheckpointSaver, MemorySaver } from "@langchain/langgraph";
import { createAgent, dynamicSystemPromptMiddleware } from "langchain";
import { ConfigurableModel } from "langchain/chat_models/universal";
import { trimMessages } from "../helpers/middlewares";
import {
  AgentContext,
  agentContextSchema,
  getExchangeRatesTool,
  getHistoricalRatesTool,
  storageKnowledgeTool,
} from "../helpers/tools";

// =============================================================================
// Cache para evitar enviar mensajes duplicados
// =============================================================================
// El streaming puede enviar el mismo mensaje varias veces en diferentes chunks.
// Este cache evita duplicados usando el id del mensaje como clave.

const sentMessages: Record<string, boolean> = {};

// =============================================================================
// CLASE AGENT: Wrapper del agente LangChain
// =============================================================================

export class Agent {
  // Handlers organizados por uuid (thread_id) para soportar múltiples conversaciones
  // Cada conversación tiene su propio array de handlers (clientes conectados)
  private messagesHandlers: Record<
    string,
    Array<(message: BaseMessage | any) => void>
  > = {};

  // Agente interno de LangChain (ReAct Agent)
  // Usamos ReturnType para inferir el tipo correcto de createAgent
  private internalAgent: ReturnType<typeof createAgent>;

  // Vector store para las búsquedas RAG (se pasa como contexto a las tools)
  private vectorStore: VectorStore;

  /**
   * Constructor del agente
   * @param model Modelo de lenguaje LangChain (Mistral, OpenAI, etc.)
   * @param vectorStore Vector store para búsquedas RAG (se inyecta en las tools)
   * @param checkpointer Persistencia de conversaciones (SQLite, Memory, etc.)
   */
  constructor(
    model: ConfigurableModel,
    vectorStore: VectorStore,
    checkpointer: BaseCheckpointSaver = new MemorySaver(),
  ) {
    this.vectorStore = vectorStore;

    // createAgent() crea un agente ReAct que puede usar herramientas
    this.internalAgent = createAgent({
      model,
      // contextSchema define la estructura del contexto que se pasa a las tools
      // Las tools acceden a él vía config.context (ej: config.context.vectorStore)
      contextSchema: agentContextSchema,
      // Herramientas disponibles para el agente
      tools: [
        storageKnowledgeTool,    // RAG sobre web storage (usa vectorStore del contexto)
        getExchangeRatesTool,    // Tasas de cambio actuales
        getHistoricalRatesTool,  // Tasas de cambio históricas
        new Calculator(),        // Calculadora matemática
      ],
      // Middleware: funciones que procesan los mensajes antes/después del LLM
      middleware: [
        trimMessages, // Limita el historial para no exceder el contexto
        // System prompt dinámico: se ejecuta en cada invocación
        // Esto permite incluir información que cambia (fecha, estado, etc.)
        dynamicSystemPromptMiddleware(() => {
          const today = new Date().toLocaleDateString("es-ES", {
            weekday: "long",
            year: "numeric",
            month: "long",
            day: "numeric",
          });

          return `Eres un asistente experto y preciso. Responde siempre en español de forma clara y concisa.

FECHA ACTUAL: ${today}

HERRAMIENTAS DISPONIBLES Y CUÁNDO USARLAS:

1. **storage_knowledge** - Base de conocimiento sobre almacenamiento web
   USAR OBLIGATORIAMENTE cuando el usuario pregunte sobre:
   - localStorage, sessionStorage, IndexedDB, cookies, Cache API
   - Cómo guardar datos en el navegador
   - Límites de almacenamiento web
   - Diferencias entre métodos de persistencia

2. **get_exchange_rates** - Tasas de cambio ACTUALES
   USAR cuando el usuario pregunte sobre:
   - Valor actual de una divisa (ej: "¿Cuánto vale el dólar hoy?")
   - Conversión de monedas al día de hoy

3. **get_historical_rates** - Tasas de cambio HISTÓRICAS
   USAR cuando el usuario pregunte sobre:
   - Valor de una divisa en una fecha pasada específica
   - Comparación de evolución de divisas entre fechas

4. **calculator** - Calculadora matemática
   USAR para cualquier cálculo numérico que requiera precisión

REGLAS:
- Si no estás seguro de la respuesta, USA las herramientas disponibles
- Para preguntas sobre storage web, SIEMPRE consulta storage_knowledge primero
- Para conversiones de moneda, USA las herramientas de tasas de cambio
- Sé conciso pero completo en tus respuestas`;
        }),
      ],
      // Persistencia de la conversación
      checkpointer,
    });
  }

  // ===========================================================================
  // messageReceived: Procesa un mensaje del usuario
  // ===========================================================================
  // Recibe un mensaje, lo pasa al agente y envía las respuestas vía streaming.

  async messageReceived(message: string, clientConfig: { thread_id: string }) {
    const { thread_id } = clientConfig;
    const initialMessage = new HumanMessage(message);

    console.log(`[Agent] Mensaje recibido en thread: ${thread_id}`);

    // stream() permite recibir respuestas incrementales del agente
    // streamMode: "values" devuelve el estado completo en cada chunk
    // context: pasa el vectorStore a las tools que lo necesiten
    const response = await this.internalAgent.stream(
      { messages: [initialMessage] },
      {
        streamMode: "values",
        configurable: { thread_id },
        // El contexto se pasa a las tools vía config.context
        context: { vectorStore: this.vectorStore } satisfies AgentContext,
      },
    );

    // Procesar cada chunk del streaming
    for await (const chunk of response) {
      if (chunk.messages) {
        chunk.messages.forEach((msg: BaseMessage) => {
          // Evitar enviar el mismo mensaje dos veces
          if (sentMessages[`${msg.id}`]) return;

          this._sendMessageToClients(thread_id, msg);
          sentMessages[`${msg.id}`] = true;
        });
      }
    }
  }

  // ===========================================================================
  // _sendMessageToClients: Notifica a todos los handlers de una conversación
  // ===========================================================================

  private _sendMessageToClients(uuid: string, message: BaseMessage | any) {
    // Verificar que existan handlers para este uuid
    if (!this.messagesHandlers[uuid]) return;
    this.messagesHandlers[uuid].forEach((handler) => handler(message));
  }

  // ===========================================================================
  // registerNewMessageHandler: Registra un handler para recibir mensajes
  // ===========================================================================
  // Se llama cuando un cliente abre una conexión SSE.

  registerNewMessageHandler(
    handler: (message: BaseMessage | any) => void,
    uuid: string,
  ) {
    this.messagesHandlers[uuid] = this.messagesHandlers[uuid] || [];
    this.messagesHandlers[uuid].push(handler);
    console.log(`[Agent] Handler registrado para thread: ${uuid}`);
  }

  // ===========================================================================
  // unregisterNewMessageHandler: Elimina un handler cuando el cliente se desconecta
  // ===========================================================================

  unregisterNewMessageHandler(
    uuid: string,
    handler: (message: BaseMessage | any) => void,
  ) {
    if (!this.messagesHandlers[uuid]) return;
    this.messagesHandlers[uuid] = this.messagesHandlers[uuid].filter(
      (h) => h !== handler,
    );
    console.log(`[Agent] Handler desregistrado para thread: ${uuid}`);
  }
}
