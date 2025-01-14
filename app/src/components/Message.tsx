import React from "react";
import Message from "../models/Message";

const MessageComponent: React.FC<{message: Message}> = ({ message}) => {
    return (
      <div className="w-full text-white px-4 flex">
        <p className="w-[5%] font-bold">{message.isAI ? "AI:" : "You:"}</p>
        <div className="w-[95%] overflow-y-auto">
          <p className="" dangerouslySetInnerHTML={{__html: message.content}}/>
        </div>
      </div>
    );
  };
  export default MessageComponent;