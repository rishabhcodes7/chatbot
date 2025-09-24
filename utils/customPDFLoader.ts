// class from LangChain — represents a chunk of text + metadata
import { Document } from "langchain/document";

import { readFile } from "fs/promises";

// Base class from LangChain for making custom document loaders
import { BaseDocumentLoader } from "langchain/document_loaders/base";

// Abstract class for loading a file/blob into memory as Buffer
export abstract class BufferLoader extends BaseDocumentLoader {
  // Takes either a file path (string) or a Blob (like from browser upload)
  constructor(public filePathOrBlob: string | Blob) {
    super(); // call parent BaseDocumentLoader constructor
  }

  // Abstract method: subclasses must define how to parse raw Buffer into Documents
  protected abstract parse(
    raw: Buffer,
    metadata: Document["metadata"]
  ): Promise<Document[]>;

  // Main method called to load and process the document
  public async load(): Promise<Document[]> {
    let buffer: Buffer;
    let metadata: Record<string, string>;

    // Case 1: If input is file path (string)
    if (typeof this.filePathOrBlob === "string") {
      buffer = await readFile(this.filePathOrBlob); // read file into Buffer
      metadata = { source: this.filePathOrBlob }; // store file path as metadata
    }
    // Case 2: If input is a Blob (like from frontend upload)
    else {
      buffer = await this.filePathOrBlob
        .arrayBuffer() // convert Blob to ArrayBuffer
        .then((ab) => Buffer.from(ab)); // convert ArrayBuffer to Node.js Buffer
      metadata = {
        // store blob info in metadata
        source: "blob",
        blobType: this.filePathOrBlob.type,
      };
    }

    // Finally, pass raw buffer + metadata to subclass's parse function
    return this.parse(buffer, metadata);
  }
}

// Custom loader specifically for PDFs
export class CustomPDFLoader extends BufferLoader {
  // Implements the abstract parse() function
  public async parse(
    raw: Buffer,
    metadata: Document["metadata"]
  ): Promise<Document[]> {
    // Dynamically import pdf-parse library (to avoid bundling debug code)
    const { pdf } = await PDFLoaderImports();

    // Use pdf-parse to extract text + metadata from PDF Buffer
    const parsed = await pdf(raw);

    // Wrap parsed content into LangChain's Document format
    return [
      new Document({
        pageContent: parsed.text, // extracted text from PDF
        metadata: {
          ...metadata, // add source info (path/blob)
          pdf_numpages: parsed.numpages, // store number of pages
        },
      }),
    ];
  }
}

// Helper function to import pdf-parse dynamically
async function PDFLoaderImports() {
  try {
    // Import pdf-parse's main function directly
    const { default: pdf } = await import("pdf-parse/lib/pdf-parse.js");
    return { pdf };
  } catch (e) {
    console.error(e);
    // If import fails, throw an error telling user to install pdf-parse
    throw new Error(
      "Failed to load pdf-parse. Please install it with eg. `npm install pdf-parse`."
    );
  }
}
