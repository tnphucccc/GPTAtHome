import React, { useState, useCallback } from "react";
import Slider from "@mui/material/Slider";
import SendIcon from "@mui/icons-material/Send";
import { usePrompt } from "../hooks/Prompt";
import Prompt from "../models/Prompt";

const SendMessage: React.FC = () => {
  const [message, setMessage] = useState<string>("");
  const [token, setToken] = useState<number>(500);
  const { setPrompt, addMsg, loading } = usePrompt();

  const valuetext = useCallback((value: number) => {
    setToken(value);
    return `${value}`;
  }, []);

  const sendMessage = useCallback(
    async (event: React.FormEvent) => {
      event.preventDefault();
      if (message.trim() === "") {
        alert("Enter a valid message");
        return;
      }
      const prompt: Prompt = {
        prompt: message,
        maxTokens: token,
      };
      setPrompt(prompt);
      const newMsg = {
        content: message,
        id: Date.now(),
        isAI: false,
      };
      addMsg(newMsg);
      setMessage("");
    },
    [message, token, setPrompt, addMsg]
  );

  return (
    <form
      className="w-full flex px-5 py-3 justify-center items-center"
      onSubmit={sendMessage}
    >
      <div className="w-1/2 py-3 rounded-lg px-5 bg-[#2f2f2f] flex">
        <div className="w-full">
          <MessageInput message={message} setMessage={setMessage} />
          <TokenSlider token={token} valuetext={valuetext} />
        </div>

        <SendButton loading={loading} />
      </div>
    </form>
  );
};

const MessageInput: React.FC<{
  message: string;
  setMessage: React.Dispatch<React.SetStateAction<string>>;
}> = React.memo(({ message, setMessage }) => (
  <textarea
    className="bg-transparent w-full focus:outline-none text-white py-2"
    placeholder="Enter your message"
    value={message}
    onChange={(e) => setMessage(e.target.value)}
  />
));

const TokenSlider: React.FC<{
  token: number;
  valuetext: (value: number) => string;
}> = React.memo(({ token, valuetext }) => (
  <div className="w-full flex">
    <p className="text-white w-1/3">Tokens: {token}</p>
    <div className="w-full flex justify-center items-center">
      <div className="w-4/5">
        <Slider
          aria-label="Token"
          defaultValue={token}
          getAriaValueText={valuetext}
          valueLabelDisplay="auto"
          shiftStep={100}
          step={100}
          marks
          min={100}
          max={2000}
        />
      </div>
    </div>
  </div>
));

const SendButton: React.FC<{ loading: boolean }> = React.memo(({ loading }) => (
  <button
    type="submit"
    className={`text-white p-2 ${
      loading ? "cursor-not-allowed opacity-50" : "cursor-pointer"
    }`}
    disabled={loading}
  >
    <SendIcon />
  </button>
));

export default SendMessage;
