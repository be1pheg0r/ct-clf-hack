import React, { useEffect, useRef } from "react";
import { Message } from "../App";
import MessageBubble from "./message";

interface ChatProps {
  messages: Message[];
}

const Chat: React.FC<ChatProps> = ({ messages }) => {
  const endRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [messages]);

  return (
    <div className="chat-window">
      {messages.map((msg) => (
        <MessageBubble key={msg.id} message={msg} />
      ))}
      <div ref={endRef} />
    </div>
  );
};

export default Chat;
