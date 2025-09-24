import React from "react";
import { Message } from "../App";
import MessageBubble from "./message";

interface ChatProps {
  messages: Message[];
}

const Chat: React.FC<ChatProps> = ({ messages }) => {
  return (
    <div className="chat-window">
      {messages.map((msg) => (
        <MessageBubble key={msg.id} message={msg} />
      ))}
    </div>
  );
};

export default Chat;
