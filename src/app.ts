import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { RetrievalQAChain } from "langchain/chains";
import { GPT4All } from 'gpt4all';
import * as dotenv from "dotenv";

dotenv.config();

const loader = new PDFLoader("src/documents/SBCA-Rules-Book.pdf");

const docs = await loader.load();

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 20,
});

// created chunks from pdf
const splittedDocs = await splitter.splitDocuments(docs);

const embeddings = new OpenAIEmbeddings();

const vectorStore = await HNSWLib.fromDocuments(
  splittedDocs,
  embeddings
);

const vectorStoreRetriever = vectorStore.asRetriever();
const model = new GPT4All({
  modelName: 'gpt-3.5-turbo'
});

const chain = RetrievalQAChain.fromLLM(model, vectorStoreRetriever);

const question = 'What is this pdf about?';

const answer = await chain.call({
  query: question
});

console.log({
  question,
  answer
});

const question1 = 'What is this pdf about?';
const answer1 = await chain.call({
  query: question1
});

console.log({
  question: question1,
  answer: answer1
});

const question2 = 'What is this pdf about?'
const answer2 = await chain.call({
  query: question2
});
console.log({
  question: question2,
  answer: answer2
});



