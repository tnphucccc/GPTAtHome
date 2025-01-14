import React, { useState, useEffect } from 'react';
import { usePrompt } from '../hooks/Prompt';

const Header: React.FC = () => {
    const [showTutorial, setShowTutorial] = useState<boolean>(true);
    const { loading, prompt } = usePrompt();

    useEffect(() => {
        if (prompt) {
            setShowTutorial(false);
        }
    }, [prompt]); // Re-run when `prompt` changes

    return (
        <div className="w-full">
            {showTutorial && (
                <p className="w-full text-center italic text-white py-1">
                    Start to give a topic and wait for our AI to generate the story!
                </p>
            )}
            {loading && (
                <div className='flex justify-center items-center text-white'>
                    <svg className="animate-spin h-5 w-5 mr-3" viewBox="0 0 24 24" fill="white"></svg>
                    <p className="text-white text-center">AI is generating the story for you...</p>
                </div>
                
            )}
        </div>
    );
};

export default Header;
