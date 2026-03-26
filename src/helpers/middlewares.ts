import {
  RemoveMessage,
  ToolMessage
} from "@langchain/core/messages";
import { REMOVE_ALL_MESSAGES } from "@langchain/langgraph";
import { createMiddleware } from "langchain";

export const trimMessages = createMiddleware({
  name: "TrimMessages",
  beforeModel: (state) => {
    const messages = state.messages;
    if (messages.length <= 3) return; // No recortar si hay pocos

    // Mantener primer mensaje + últimos 4
    const firstMsg = messages[0];
    let recentMsgs = messages.slice(-4);

    // Asegurar que no empezamos con un mensaje 'tool' huérfano.
    // Si el primer mensaje reciente es de tipo 'tool', necesitamos incluir
    // el mensaje 'assistant' con tool_calls que lo precede.
    while (recentMsgs.length > 0 && ToolMessage.isInstance(recentMsgs[0])) {
      // Buscar el índice en el array original
      const idx = messages.indexOf(recentMsgs[0]);
      if (idx > 0) {
        // Incluir el mensaje anterior (debería ser assistant con tool_calls)
        recentMsgs = [messages[idx - 1], ...recentMsgs];
      } else {
        // No hay mensaje anterior, eliminar el tool huérfano
        recentMsgs = recentMsgs.slice(1);
      }
    }

    // También asegurar que no empezamos con un 'assistant' sin contexto previo
    // después del primer mensaje (esto es menos problemático pero más limpio)
    const newMessages = [firstMsg, ...recentMsgs];

    return {
      messages: [
        new RemoveMessage({ id: REMOVE_ALL_MESSAGES }),
        ...newMessages,
      ],
    };
  },
});
