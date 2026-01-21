import { RemoveMessage } from "@langchain/core/messages";
import { REMOVE_ALL_MESSAGES } from "@langchain/langgraph";
import { createMiddleware } from "langchain";

export const trimMessages = createMiddleware({
  name: "TrimMessages",
  beforeModel: (state) => {
    const messages = state.messages;
    if (messages.length <= 3) return; // No recortar si hay pocos

    // Mantener primer mensaje + últimos 4
    const firstMsg = messages[0];
    const recentMsgs = messages.slice(-4);
    const newMessages = [firstMsg, ...recentMsgs];

    return {
      messages: [
        new RemoveMessage({ id: REMOVE_ALL_MESSAGES }),
        ...newMessages,
      ],
    };
  },
});
