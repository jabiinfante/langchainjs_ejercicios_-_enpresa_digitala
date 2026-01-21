/**
 * =============================================================================
 * EJERCICIO: Servidor Web con Agente LangChain y Streaming (SSE)
 * =============================================================================
 *
 * Este ejercicio demuestra cómo integrar un agente de LangChain con un servidor
 * web Fastify, usando Server-Sent Events (SSE) para streaming en tiempo real.
 *
 * ARQUITECTURA:
 * ┌─────────────┐    POST /message     ┌─────────────┐
 * │   Cliente   │ ─────────────────────▶│   Fastify   │
 * │  (Browser)  │                       │   Server    │
 * │             │◀───────────────────── │             │
 * └─────────────┘    SSE /stream        └──────┬──────┘
 *                                              │
 *                                              ▼
 *                                       ┌─────────────┐
 *                                       │   Agent     │
 *                                       │ (LangChain) │
 *                                       └─────────────┘
 *
 * CONCEPTOS CLAVE:
 * - SSE (Server-Sent Events): Protocolo para enviar datos del servidor al cliente
 *   en tiempo real, ideal para streaming de respuestas de LLMs
 * - Checkpointer: Guarda el estado de la conversación (memoria persistente)
 * - thread_id: Identificador único de conversación para mantener contexto
 *
 * FLUJO:
 * 1. Cliente abre conexión SSE en GET /stream (recibe mensajes en tiempo real)
 * 2. Cliente envía mensaje POST /message
 * 3. Servidor pasa mensaje al Agente
 * 4. Agente procesa y genera respuesta (puede usar tools)
 * 5. Cada chunk de respuesta se envía al cliente vía SSE
 *
 * DEPENDENCIAS:
 * - fastify: Framework web rápido y de bajo consumo para Node.js
 * - fastify-sse-v2: Plugin para manejar Server-Sent Events
 * - @fastify/cors: Plugin para habilitar CORS
 * =============================================================================
 */

import cors from "@fastify/cors";
import { SqliteSaver } from "@langchain/langgraph-checkpoint-sqlite";
import { MistralAIEmbeddings } from "@langchain/mistralai";
import { QdrantVectorStore } from "@langchain/qdrant";
import Fastify from "fastify";
import { FastifySSEPlugin } from "fastify-sse-v2";
import { initChatModel } from "langchain";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Agent } from "./agents_wrapper/agent";

// =============================================================================
// PASO 1: Inicializar el Agente con memoria persistente
// =============================================================================
// El agente se inicializa ANTES del servidor para que esté listo cuando
// lleguen las peticiones. SqliteSaver guarda el historial de conversaciones.

const model = await initChatModel("mistral:mistral-large-latest", {
  timeout: 120000, // 2 minutos de timeout para respuestas largas
});

const embeddings = new MistralAIEmbeddings({
  model: "mistral-embed",
});

const vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, {
  url: process.env.QDRANT_URL,
  collectionName: "langchainjs-testing-dia2",
  apiKey: process.env.QDRANT_API_KEY,
});
// ó usar inMemorry vector store para pruebas rápidas
// import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
// new MemoryVectorStore(embeddings)

// Checkpointer: Guarda el estado de las conversaciones en SQLite
// Permite que las conversaciones persistan entre reinicios del servidor
const checkpointer = SqliteSaver.fromConnString(
  join(tmpdir(), "agent_memory.db"),
);
const agent = new Agent(model, vectorStore, checkpointer);

// =============================================================================
// PASO 2: Configurar el servidor Fastify
// =============================================================================

const fastify = Fastify({
  logger: true, // Habilitar logs para debugging
});

// CORS: Permitir peticiones desde cualquier origen (desarrollo)
// En producción, restringir a dominios específicos
fastify.register(cors, {
  origin: "*",
});

// Plugin SSE: Habilita el método reply.sse() para streaming
fastify.register(FastifySSEPlugin);

// =============================================================================
// PASO 3: Ruta de health check
// =============================================================================

fastify.get("/", async (request, reply) => {
  return { message: "Hola agente!", status: "running" };
});

// =============================================================================
// PASO 4: Ruta POST /message - Recibir mensajes del cliente
// =============================================================================
// El cliente envía un mensaje y el servidor lo pasa al agente.
// La respuesta se envía de forma asíncrona vía SSE (no en esta ruta).

fastify.post<{ Params: { uuid?: string }; Body: { message: string } }>(
  "/message",
  async (request, reply) => {
    const { message } = request.body;

    // thread_id identifica la conversación
    // En producción: obtener de cookies, headers, JWT, etc.
    const thread_id = request.params.uuid || "chat-id-XXX";

    // Enviar mensaje al agente (procesamiento asíncrono)
    // Las respuestas se enviarán vía SSE a los clientes suscritos
    agent.messageReceived(message, { thread_id });

    // Respuesta inmediata: confirmar que el mensaje fue recibido
    return { status: "Mensaje recibido", thread_id };
  },
);

// =============================================================================
// PASO 5: Ruta GET /stream - Conexión SSE para recibir respuestas
// =============================================================================
// El cliente mantiene una conexión abierta y recibe mensajes en tiempo real.
// Documentación SSE: https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events

fastify.get<{ Params: { uuid?: string } }>(
  "/stream",
  async (request, reply) => {
    const thread_id = request.params.uuid || "chat-id-XXX";

    // Enviar evento de conexión establecida
    reply.sse({
      data: JSON.stringify({ connected: true }),
      event: "connected",
    });

    // Handler: Se ejecuta cada vez que el agente genera un nuevo mensaje
    const handler = (message: any) => {
      reply.sse({ data: JSON.stringify(message), event: "message" });
    };

    // Registrar el handler para esta conversación (thread_id)
    agent.registerNewMessageHandler(handler, thread_id);

    // Cleanup: Desregistrar cuando el cliente cierra la conexión
    reply.raw.on("close", () => {
      request.log.info(`SSE connection closed for thread: ${thread_id}`);
      agent.unregisterNewMessageHandler(thread_id, handler);
    });
  },
);

// =============================================================================
// PASO 6: Arrancar el servidor
// =============================================================================

try {
  await fastify.listen({ port: 3000, host: "0.0.0.0" });
  console.log("Servidor corriendo en http://localhost:3000");
  console.log("Endpoints disponibles:");
  console.log("  GET  /        - Health check");
  console.log("  POST /message - Enviar mensaje al agente");
  console.log("  GET  /stream  - Conexión SSE para recibir respuestas");
} catch (err) {
  fastify.log.error(err);
  process.exit(1);
}
