import React, { useEffect, useRef } from "react";
import MessageComponent from "./Message";
import { usePrompt } from "../hooks/Prompt";
import SendMessage from "./SendMessage";


const ChatBox: React.FC = () => {
  const { messages } = usePrompt();
  const messagesBottomDiv = useRef<HTMLDivElement | null>(null);
  const scrollToBottom = () => {
    if (!messagesBottomDiv || !messagesBottomDiv.current) return;
    messagesBottomDiv.current.scrollTop =
      messagesBottomDiv.current?.scrollHeight;
  };
  useEffect(() => {
    scrollToBottom();
  }, [messages]);
  return (
    <div className="flex flex-col h-[90vh]">
      <div className="w-full grow overflow-y-auto" ref={messagesBottomDiv}>
        {messages?.map((message) => (
          <>
            <MessageComponent key={message.id} message={message} />
            <br />
          </>
          
        ))}
        
      </div>
      <div className="w-full py-2 flex-none">
        <SendMessage />
      </div>
    </div>
  );
};


export default ChatBox;