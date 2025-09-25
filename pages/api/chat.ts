import type { NextApiRequest, NextApiResponse } from "next";
import { Document } from "langchain/document";
import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from "@langchain/google-genai";
import { PineconeStore } from "@langchain/pinecone";
import { combineDocumentsFn } from "@/utils/makechain";
import { pinecone } from "@/utils/pinecone-client";
import { PINECONE_INDEX_NAME, PINECONE_NAME_SPACE } from "@/config/pinecone";
import { HumanMessage } from "@langchain/core/messages";
import puppeteer from "puppeteer";

// URLs to crawl if Pinecone fails
const urlsToVisit = ["https://radheshrinivasafoundation.com/"];

export async function crawlWebsite(
  startUrl: string,
  maxPages = 200
): Promise<string[]> {
  const browser = await puppeteer.launch({
    headless: true,
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
  });

  const visited = new Set<string>();
  const toVisit: string[] = [startUrl];
  const origin = new URL(startUrl).origin;

  const normalizeUrl = (url: string) => {
    try {
      const u = new URL(url, origin);
      return u.href.split("#")[0]; // keep query params but strip hashes
    } catch {
      return null;
    }
  };

  while (toVisit.length > 0 && visited.size < maxPages) {
    const url = toVisit.shift()!;
    const normalizedUrl = normalizeUrl(url);
    if (!normalizedUrl || visited.has(normalizedUrl)) continue;

    try {
      const page = await browser.newPage();
      page.setDefaultNavigationTimeout(60000);
      await page.goto(normalizedUrl, { waitUntil: "networkidle2" });

      // Extract ALL links as absolute
      const links: string[] = await page.evaluate(() =>
        Array.from(document.querySelectorAll("a[href]"))
          .map((a) => (a as HTMLAnchorElement).href)
          .filter(
            (href) => href.startsWith("http") && !href.startsWith("javascript:")
          )
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
      const page = await browser.newPage();
      page.setDefaultNavigationTimeout(60000);
      await page.goto(url, { waitUntil: "networkidle2" });

      const pageText = await page.evaluate(() => {
        const main = document.querySelector("main");
        return main ? main.innerText : document.body.innerText || "";
      });

      const cleanedText = pageText.replace(/\s+/g, " ").trim();

      const chunkSize = 1000;
      const overlap = 200;

      for (let i = 0; i < cleanedText.length; i += chunkSize - overlap) {
        const chunk = cleanedText.substring(i, i + chunkSize);
        if (chunk.length > 100) {
          documents.push(
            new Document({
              pageContent: chunk,
              metadata: { source: url, chunkIndex: i, type: "web_content" },
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
    // Pinecone setup
    const index = pinecone.Index(PINECONE_INDEX_NAME);
    const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GEMINI_API_KEY!,
      model: "text-embedding-004",
    });

    const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
      pineconeIndex: index,
      textKey: "text",
      namespace: PINECONE_NAME_SPACE,
    });

    const retriever = vectorStore.asRetriever();
    const pineconeDocs = await retriever.invoke(sanitizedQuestion);

    // Score and filter
    const scoredDocs = pineconeDocs.map((doc) => {
      const questionWords = sanitizedQuestion.toLowerCase().split(/\s+/);
      const content = doc.pageContent.toLowerCase();
      const score = questionWords.filter(
        (w) => content.includes(w) && w.length > 1
      ).length;
      return { doc, score };
    });

    const relevantDocs = scoredDocs
      .filter((d) => d.score > 0)
      .map((d) => d.doc);

    const contextText =
      relevantDocs.length > 0 ? combineDocumentsFn(relevantDocs) : "";

    // Combine past messages
    const pastMessages = history
      .map(([q, a]) => `Human: ${q}\nAssistant: ${a}`)
      .join("\n\n");

    const model = new ChatGoogleGenerativeAI({
      apiKey: process.env.GOOGLE_API_KEY,
      model: "gemini-2.5-flash",
      temperature: 0.1,
    });

    // Attempt answer from Pinecone context
    let answer = "";
    if (contextText) {
      const prompt = `
You are a helpful AI assistant. Use the following context and conversation history to answer the question. 
Do not say "I don't know" and do not mention the source explicitly. Resolve pronouns like "it" using conversation history. Use your general knowledge to answer the question, if the answer is not found in pinecone documents or in website crwaling.Go to the website crawling if you don't have answer to the question. Never say I don't know or similar words.Make the answer empty string if you don't know the answer.

Context:
${contextText}

Conversation history:
${pastMessages || "No prior history."}

Current question:
Human: ${sanitizedQuestion}
Assistant:
`;
      const raw = await model.invoke([new HumanMessage(prompt)]);
      answer = typeof raw === "string" ? raw : (raw?.text ?? "");
    }

    // If Pinecone answer is weak, crawl website
    if (!answer) {
      console.log("Pinecone answer insufficient, crawling website...");
      const allUrls: string[] = [];
      for (const domain of urlsToVisit) {
        const domainUrls = await crawlWebsite(domain, 50);
        allUrls.push(...domainUrls);
      }

      const webDocs = await fetchMultiplePages(allUrls);
      console.log("Web Docs: ", webDocs);
      const webContext = combineDocumentsFn(webDocs);

      const prompt = `
You are a helpful AI assistant. Use the following web content and conversation history to answer the question. 
Do not say "I don't know" and do not mention the source explicitly. Resolve pronouns like "it" using conversation history.

Context:
${webContext}

Conversation history:
${pastMessages || "No prior history."}

Current question:
Human: ${sanitizedQuestion}
Assistant:
`;

      const raw = await model.invoke([new HumanMessage(prompt)]);
      answer = typeof raw === "string" ? raw : (raw?.text ?? "");
    }

    res.status(200).json({
      text: answer,
      sourceDocuments: relevantDocs.slice(0, 5),
    });
  } catch (err: any) {
    console.error("Error in handler:", err);
    res.status(500).json({ error: err.message || "Something went wrong" });
  }
}
