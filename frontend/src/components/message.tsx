import React from "react";
import { Message } from "../App";
import ResultCard from "./ResultCard";
import { ResultData } from "../types";

interface MessageProps {
  message: Message;
}

const MessageBubble: React.FC<MessageProps> = ({ message }) => {
  if (message.type === "result") {
    return (
      <div className="message-row system-row">
        <div style={{ width: 12 }} />
        <div className="bubble bubble-system" style={{ padding: 10 }}>
          <ResultCard data={message.content as ResultData} />
        </div>
      </div>
    );
  }

  const isUser = message.type === "user";
  const text = message.content as string; // ✅ гарантируем строку

  return (
    <div className={`message-row ${isUser ? "user-row" : "system-row"}`}>
      {!isUser && <div className="avatar avatar-system">AI</div>}
      <div className={`bubble ${isUser ? "bubble-user" : "bubble-system"}`}>
        <div className="bubble-content">{text}</div>
      </div>
      {isUser && <div className="avatar avatar-user">Вы</div>}
    </div>
  );
};

export default MessageBubble;
