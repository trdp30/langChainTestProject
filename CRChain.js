/* Conversational Retrieval Chain */

import "dotenv/config";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import { MessagesPlaceholder } from "@langchain/core/prompts";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";

const init = async () => {
  const chatModel = new ChatOpenAI({
    openAIApiKey: process.env.OPENAI_API_KEY,
  });

  const embeddings = new OpenAIEmbeddings({
    openAIApiKey: process.env.OPENAI_API_KEY,
  });

  const loader = new CheerioWebBaseLoader(
    "https://docs.smith.langchain.com/user_guide"
  );

  const docs = await loader.load();
  const splitter = new RecursiveCharacterTextSplitter();

  const splitDocs = await splitter.splitDocuments(docs);

  const vectorstore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
  );

  const retriever = vectorstore.asRetriever();

  // const historyAwarePrompt = ChatPromptTemplate.fromMessages([
  //   new MessagesPlaceholder("chat_history"),
  //   ["user", "{input}"],
  //   [
  //     "user",
  //     "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
  //   ],
  // ]);

  // const historyAwareRetrieverChain = await createHistoryAwareRetriever({
  //   llm: chatModel,
  //   retriever,
  //   rephrasePrompt: historyAwarePrompt,
  // });

  // const chatHistory = [
  //   new HumanMessage("Can LangSmith help test my LLM applications?"),
  //   new AIMessage("Yes!"),
  // ];

  // const response = await historyAwareRetrieverChain.invoke({
  //   chat_history: chatHistory,
  //   input: "Tell me how!",
  // });

  const historyAwarePrompt = ChatPromptTemplate.fromMessages([
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
    [
      "user",
      "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
    ],
  ]);

  const historyAwareRetrieverChain = await createHistoryAwareRetriever({
    llm: chatModel,
    retriever,
    rephrasePrompt: historyAwarePrompt,
  });

  const historyAwareRetrievalPrompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "Answer the user's questions based on the below context:\n\n{context}",
    ],
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
  ]);

  const historyAwareCombineDocsChain = await createStuffDocumentsChain({
    llm: chatModel,
    prompt: historyAwareRetrievalPrompt,
  });

  const conversationalRetrievalChain = await createRetrievalChain({
    retriever: historyAwareRetrieverChain,
    combineDocsChain: historyAwareCombineDocsChain,
  });

  const response = await conversationalRetrievalChain.invoke({
    chat_history: [
      new HumanMessage("Can LangSmith help test my LLM applications?"),
      new AIMessage("Yes!"),
    ],
    input: "tell me how",
  });

  console.log(response.answer);
};

init();
