import Answer from '../models/Answer';
import Message from '../models/Message';
const fromAnswer = (answer: Answer): Message => {
    const rawContent = answer.response;
    const content = (rawContent.replace(/^[^.!?]*[.!?]\s*/, '')).replace(/\n/g, '<br />');
    return{
        content,
        id: Date.now(),
        isAI: true,
    }

}
const MessageService = {
    fromAnswer
};
export default MessageService;