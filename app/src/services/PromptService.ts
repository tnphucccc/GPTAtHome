import Answer from "../models/Answer";
import Prompt from "../models/Prompt";
import axios, { AxiosRequestConfig } from 'axios';
import { Err, Ok, Result } from "../types";
const baseURL = "http://localhost:8000"
// const baseURL = "http://server:8000"
const config: AxiosRequestConfig = {
    headers: {
        'Content-Type': 'application/json',
        "accept": "application/json"
    }
}

const requestHandler = axios.create({
    baseURL: baseURL
});
const post = async (prompt: Prompt): Promise<Result<Answer>> => {
    console.log(baseURL);
    const serializedPrompt = toJson(prompt);
    const response = await requestHandler.post<Answer>("/generate", serializedPrompt, config);
    if (response.status !== 200) {
        return Err(new Error("Failed to fetch data"));
    }
    return Ok(response.data);
}

const toJson = (prompt: Prompt) => {
    return JSON.stringify({
        prompt: prompt.prompt,
        max_tokens: prompt.maxTokens
    })
}

const PromptService = {
    post
};
export default PromptService;