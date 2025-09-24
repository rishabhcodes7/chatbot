/**
 * Change the namespace to the namespace on Pinecone you'd like to store your embeddings.
 */

if (!process.env.PINECONE_INDEX_NAME) {
  throw new Error("Missing Pinecone index name in .env file");
}

const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX_NAME ?? "";
const GEMINI_API_KEY = process.env.GEMINI_API_KEY ?? "";

const PINECONE_NAME_SPACE = "my-docs"; //namespace is optional for your vectors

export { PINECONE_INDEX_NAME, GEMINI_API_KEY, PINECONE_NAME_SPACE };
