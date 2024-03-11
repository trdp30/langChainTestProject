/* Agent */

import "dotenv/config";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { createRetrieverTool } from "langchain/tools/retriever";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { pull } from "langchain/hub";
import { createOpenAIFunctionsAgent, AgentExecutor } from "langchain/agents";
import { HumanMessage, AIMessage } from "@langchain/core/messages";

const init = async () => {
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

  const retrieverTool = await createRetrieverTool(retriever, {
    name: "langsmith_search",
    description:
      "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
  });

  const searchTool = new TavilySearchResults();

  const tools = [retrieverTool, searchTool];

  const agentPrompt = await pull("hwchase17/openai-functions-agent");

  const chatModel = new ChatOpenAI({
    openAIApiKey: process.env.OPENAI_API_KEY,
    modelName: process.env.MODEL,
    temperature: 0,
  });

  const agent = await createOpenAIFunctionsAgent({
    llm: chatModel,
    tools,
    prompt: agentPrompt,
  });

  const agentExecutor = new AgentExecutor({
    agent,
    tools,
    verbose: false,
  });

  const agentResult = await agentExecutor.invoke({
    input: "how can LangSmith help with testing?",
  });

  console.log("agentResult.output: ", agentResult.output);

  const agentResult2 = await agentExecutor.invoke({
    input: "what is the weather in SF?",
  });

  console.log("agentResult2.output: ", agentResult2.output);

  const agentResult3 = await agentExecutor.invoke({
    chat_history: [
      new HumanMessage("Can LangSmith help test my LLM applications?"),
      new AIMessage("Yes!"),
    ],
    input: "Tell me how",
  });

  console.log("agentResult3.output: ", agentResult3.output);
};

init();
