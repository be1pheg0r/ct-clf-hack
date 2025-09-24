import React from "react";
import { Message } from "../App";

interface MessageProps {
  message: Message;
}

const MessageBubble: React.FC<MessageProps> = ({ message }) => {
  return (
    <div className={`message ${message.type}`}>
      <span>{message.content}</span>
    </div>
  );
};

export default MessageBubble;
