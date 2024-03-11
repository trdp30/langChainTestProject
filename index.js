import "dotenv/config";
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";

const openAIInitialization = async () => {
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

  const prompt = ChatPromptTemplate.fromTemplate(
    `Answer the following question based only on the provided context: <context> {context} </context> Question: {input}`
  );

  const documentChain = await createStuffDocumentsChain({
    llm: chatModel,
    prompt,
  });

  const retriever = vectorstore.asRetriever();

  const retrievalChain = await createRetrievalChain({
    combineDocsChain: documentChain,
    retriever,
  });

  const result = await retrievalChain.invoke({
    input: "what is LangSmith?",
  });

  console.log(result.answer);
};

openAIInitialization();
