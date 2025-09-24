import type { NextApiRequest, NextApiResponse } from "next";
import { Document } from "langchain/document";
import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from "@langchain/google-genai";
import { PineconeStore } from "@langchain/pinecone";
import { combineDocumentsFn, makeChain } from "@/utils/makechain";
import { pinecone } from "@/utils/pinecone-client";
import { PINECONE_INDEX_NAME, PINECONE_NAME_SPACE } from "@/config/pinecone";
import { HumanMessage } from "@langchain/core/messages";
import puppeteer from "puppeteer";
import { Crawl4AI } from "crawl4ai";

// Hardcoded URLs to fetch content from
const urlsToVisit = ["https://radheshrinivasafoundation.com/"];

export async function crawlWebsite(
  startUrl: string,
  maxPages = 50
): Promise<string[]> {
  const browser = await puppeteer.launch({
    headless: true,
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
  });

  const visited = new Set<string>();
  const toVisit: string[] = [startUrl];
  const origin = new URL(startUrl).origin;

  function normalizeUrl(url: string) {
    try {
      const u = new URL(url, origin);
      return u.origin + u.pathname; // remove hashes & query params
    } catch {
      return null;
    }
  }

  while (toVisit.length > 0 && visited.size < maxPages) {
    const url = toVisit.shift()!;
    const normalizedUrl = normalizeUrl(url);
    if (!normalizedUrl || visited.has(normalizedUrl)) continue;

    try {
      const page = await browser.newPage();
      page.setDefaultNavigationTimeout(60000);
      await page.goto(url, { waitUntil: "networkidle2" });

      // Wait a bit for SPA JS to render links
      await new Promise((resolve) => setTimeout(resolve, 2000));

      // Extract all links
      const links: string[] = await page.evaluate(() =>
        Array.from(document.querySelectorAll("a[href]"))
          .map((a) => a.getAttribute("href") || "")
          .filter((href) => !!href)
      );

      for (const link of links) {
        const nUrl = normalizeUrl(link);
        if (nUrl && nUrl.startsWith(origin) && !visited.has(nUrl)) {
          toVisit.push(nUrl);
        }
      }

      visited.add(normalizedUrl);
      await page.close();
    } catch (err) {
      console.error("Failed to crawl URL:", url, err);
    }
  }

  await browser.close();
  return Array.from(visited);
}

export async function fetchMultiplePages(urls: string[]): Promise<Document[]> {
  const browser = await puppeteer.launch({
    headless: true,
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
  });

  const documents: Document[] = [];

  for (const url of urls) {
    try {
      console.log("Visiting:", url);
      const page = await browser.newPage();
      page.setDefaultNavigationTimeout(60000);

      await page.goto(url, { waitUntil: "networkidle2" });

      const pageText = await page.evaluate(() => {
        const main = document.querySelector("main");
        return main ? main.innerText : document.body.innerText || "";
      });

      // Clean and chunk the text for better retrieval
      const cleanedText = pageText.replace(/\s+/g, " ").trim();

      // Create multiple documents by chunking the content
      const chunkSize = 1000; // characters per chunk
      const overlap = 200; // overlap between chunks

      for (let i = 0; i < cleanedText.length; i += chunkSize - overlap) {
        const chunk = cleanedText.substring(i, i + chunkSize);
        if (chunk.length > 100) {
          // Only include substantial chunks
          documents.push(
            new Document({
              pageContent: chunk,
              metadata: {
                source: url,
                chunkIndex: i,
                type: "web_content",
              },
            })
          );
        }
      }

      await page.close();
    } catch (err) {
      console.error("Failed to fetch URL:", url, err);
    }
  }

  await browser.close();
  console.log("Total document chunks created:", documents.length);
  return documents;
}
export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  const { question, history } = req.body;

  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }

  if (!question) {
    return res.status(400).json({ message: "No question in the request" });
  }

  const sanitizedQuestion = question.trim().replaceAll("\n", " ");

  try {
    const index = pinecone.Index(PINECONE_INDEX_NAME);
    const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GEMINI_API_KEY!,
      model: "text-embedding-004",
    });

    // Create vector store
    const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
      pineconeIndex: index,
      textKey: "text",
      namespace: PINECONE_NAME_SPACE,
    });

    // Capture retrieved documents
    let resolveWithDocuments: (value: Document[]) => void;
    const documentPromise = new Promise<Document[]>((resolve) => {
      resolveWithDocuments = resolve;
    });

    const retriever = vectorStore.asRetriever({
      callbacks: [
        {
          handleRetrieverEnd(documents) {
            resolveWithDocuments(documents);
          },
        },
      ],
    });

    const pineconeDocs = await retriever.invoke(sanitizedQuestion);

    // Compute a relevance score for each document
    const scoredDocs = pineconeDocs.map((doc) => {
      const questionWords = sanitizedQuestion.toLowerCase().split(/\s+/);
      const content = doc.pageContent.toLowerCase();

      const score = questionWords.filter(
        (word) => content.includes(word) && word.length > 1
      ).length;

      return { doc, score };
    });

    // Filter only highly relevant docs
    const relevantDocs = scoredDocs
      .filter((d) => d.score > 1) // you can tweak the threshold
      .map((d) => d.doc);

    console.log(`Found ${relevantDocs.length} relevant Pinecone documents`);

    if (relevantDocs.length === 0) {
      // Pinecone didn't have good answers, so crawl the website
      console.log("No relevant Pinecone docs found, crawling website...");

      const allUrls: string[] = [];
      for (const domain of urlsToVisit) {
        const domainUrls = await crawlWebsite(domain, 50); // max 50 pages per domain
        allUrls.push(...domainUrls);
      }

      const webDocuments = await fetchMultiplePages(allUrls);

      // Optional: do the same relevance scoring for web docs
      const scoredWebDocs = webDocuments.map((doc) => {
        const questionWords = sanitizedQuestion.toLowerCase().split(/\s+/);
        const content = doc.pageContent.toLowerCase();

        const score = questionWords.filter(
          (word) => content.includes(word) && word.length > 3
        ).length;

        return { doc, score };
      });

      relevantDocs.push(
        ...scoredWebDocs.filter((d) => d.score > 1).map((d) => d.doc)
      );

      console.log(`Found ${relevantDocs.length} relevant web documents`);
    }

    let contextText =
      relevantDocs.length > 0 ? combineDocumentsFn(relevantDocs) : "";

    // Use the chain with the combined context
    const chain = makeChain(retriever);

    const pastMessages = history
      .map((message: [string, string]) =>
        [`Human: ${message[0]}`, `Assistant: ${message[1]}`].join("\n")
      )
      .join("\n");
    let response: string;
    if (contextText) {
      // Force the model to use the provided context
      const enhancedPrompt = `You are a helpful AI assistant. Use the following context and conversation history to answer the question. Do not say "I don't know" and do not mention the source explicitly. 

Past conversation:
${pastMessages || "No prior history."}
<context>
${contextText}
</context>

Human: ${sanitizedQuestion}

Answer:`;

      const model = new ChatGoogleGenerativeAI({
        apiKey: process.env.GOOGLE_API_KEY,
        model: "gemini-2.5-flash",
        temperature: 0.1, // Lower temperature for more factual responses
      });

      const rawResponse = await model.invoke([
        new HumanMessage(enhancedPrompt),
      ]);
      response =
        typeof rawResponse === "string"
          ? rawResponse
          : (rawResponse?.text ?? "");
    } else {
      // Fallback to general knowledge
      const model = new ChatGoogleGenerativeAI({
        apiKey: process.env.GOOGLE_API_KEY,
        model: "gemini-2.5-flash",
        temperature: 0.2,
      });

      const rawResponse = await model.invoke([
        new HumanMessage(sanitizedQuestion),
      ]);
      response =
        typeof rawResponse === "string"
          ? rawResponse
          : (rawResponse?.text ?? "");
    }

    const sourceDocuments = await documentPromise;

    res.status(200).json({
      text: response,
      sourceDocuments: relevantDocs.slice(0, 5), // Return top 5 relevant docs
    });
  } catch (error: any) {
    console.error("Error in handler:", error);
    res.status(500).json({ error: error.message || "Something went wrong" });
  }
}
