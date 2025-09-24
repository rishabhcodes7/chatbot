import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import type { Document } from "langchain/document";
import type { BaseRetriever } from "@langchain/core/retrievers";
import { RunnableLambda } from "@langchain/core/runnables";

// Prompt for rephrasing follow-up into standalone question
const CONDENSE_TEMPLATE = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

<chat_history>
{chat_history}
</chat_history>

Follow Up Input: {question}
Standalone question:`;

const QA_TEMPLATE = `You are an expert AI assistant.
Use the provided context to answer the question if it is relevant.
If the context does not contain the answer, answer fully from your own knowledge.
Never say "I am not sure" or "I don't know".
Never mention the context or chat history explicitly in your answer.

<context>
{context}
</context>

<chat_history>
{chat_history}
</chat_history>

Question: {question}

Answer in markdown, directly and naturally without phrases like 
"Based on the provided context" or similar introductions:`;

// Combine multiple docs into one string
export const combineDocumentsFn = (docs: Document[], separator = "\n\n") => {
  return docs.map((doc) => doc.pageContent).join(separator);
};

// Build the full chain
export const makeChain = (retriever: BaseRetriever) => {
  const condenseQuestionPrompt =
    ChatPromptTemplate.fromTemplate(CONDENSE_TEMPLATE);
  const answerPrompt = ChatPromptTemplate.fromTemplate(QA_TEMPLATE);

  const model = new ChatGoogleGenerativeAI({
    apiKey: process.env.GOOGLE_API_KEY,
    model: "gemini-2.5-flash",
    temperature: 0.2,
  });

  // Rewrite follow-up into standalone question
  const standaloneQuestionChain = RunnableSequence.from([
    condenseQuestionPrompt, // already outputs a PromptValue
    model, // model accepts PromptValue directly
    new StringOutputParser(),
  ]);

  // Combine retrieved documents
  const retrievalChain = retriever.pipe(
    RunnableLambda.from((docs: Document[]) => combineDocumentsFn(docs))
  );

  // Answer using context if relevant, otherwise general knowledge
  const answerChain = RunnableSequence.from([
    {
      context: RunnableSequence.from([
        (input) => input.question,
        retrievalChain,
      ]),
      chat_history: (input) => input.chat_history,
      question: (input) => input.question,
    },
    answerPrompt,
    model,
    new StringOutputParser(),
  ]);

  // Full conversational chain
  const conversationalRetrievalQAChain = RunnableSequence.from([
    {
      question: standaloneQuestionChain,
      chat_history: (input) => input.chat_history,
    },
    answerChain,
  ]);

  return conversationalRetrievalQAChain;
};
