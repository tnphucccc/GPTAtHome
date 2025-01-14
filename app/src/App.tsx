import "./App.css";
import ChatBox from "./components/ChatBox"
import { PromptProvider } from "./hooks/Prompt"
import Header from "./components/Header";

function App() {


  return (
    <div className="w-full h-screen bg-[#212121] flex flex-col">
      <p className="w-full text-center text-white text-4xl py-2">Shakespeare GPT</p>
      <PromptProvider>
        <Header />
        <div className="w-full flex-grow">
          <ChatBox />
        </div>
      </PromptProvider>
    </div>

  )
}

export default App
