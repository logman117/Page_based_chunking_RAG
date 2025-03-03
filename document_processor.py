import os
import fitz  # PyMuPDF
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    page_number: int
    document_id: str
    chunk_id: str
    metadata: Dict = None

class DocumentProcessor:
    """Processes PDF documents for a RAG system with page-anchored chunking."""
    
    def __init__(self, 
                 max_chunk_size: int = 1000, 
                 chunk_overlap: int = 100):
        """
        Initialize the document processor.
        
        Args:
            max_chunk_size: Maximum number of characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process_pdf(self, pdf_path: str, document_id: str) -> List[Chunk]:
        """
        Process a PDF document and return a list of chunks with metadata.
        
        Args:
            pdf_path: Path to the PDF file
            document_id: Unique identifier for the document
            
        Returns:
            List of Chunk objects
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Open the PDF
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.error(f"Error opening PDF {pdf_path}: {e}")
            raise
        
        chunks = []
        
        # Process each page
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_number = page_idx + 1  # 1-indexed page numbers
            
            # Extract text from the page
            text = page.get_text()
            
            # Clean and normalize text
            text = self._clean_text(text)
            
            # Skip if page is empty after cleaning
            if not text.strip():
                logger.warning(f"Page {page_number} is empty after cleaning")
                continue
            
            # Extract any page-specific metadata (headers, footers, etc.)
            metadata = self._extract_page_metadata(text, page_number)
            
            # Chunk the page text
            page_chunks = self._chunk_text(text, page_number, document_id, metadata)
            chunks.extend(page_chunks)
            
            logger.info(f"Processed page {page_number}: created {len(page_chunks)} chunks")
        
        logger.info(f"Completed processing {pdf_path}: created {len(chunks)} total chunks")
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers and headers/footers (customize as needed)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        return text.strip()
    
    def _extract_page_metadata(self, text: str, page_number: int) -> Dict:
        """
        Extract metadata from page text.
        
        This can be customized to extract headers, section titles, etc.
        """
        metadata = {
            "page_number": page_number,
        }
        
        # Extract section headers (customize regex as needed)
        section_match = re.search(r'^(Chapter|Section)\s+\d+[.:]\s+(.+?)$', 
                                 text, re.MULTILINE)
        if section_match:
            metadata["section"] = section_match.group(2).strip()
        
        return metadata
    
    def _chunk_text(self, 
                   text: str, 
                   page_number: int, 
                   document_id: str,
                   metadata: Dict) -> List[Chunk]:
        """
        Split page text into chunks while preserving page integrity.
        
        Args:
            text: Text content of the page
            page_number: Page number
            document_id: Document identifier
            metadata: Page-level metadata
            
        Returns:
            List of Chunk objects for this page
        """
        chunks = []
        
        # If text is shorter than max_chunk_size, keep it as one chunk
        if len(text) <= self.max_chunk_size:
            chunk = Chunk(
                text=text,
                page_number=page_number,
                document_id=document_id,
                chunk_id=f"{document_id}_p{page_number}_c1",
                metadata=metadata
            )
            return [chunk]
        
        # Split text into sentences to use as boundaries for chunks
        sentences = self._split_into_sentences(text)
        
        current_chunk = ""
        chunk_count = 1
        
        for sentence in sentences:
            # If adding this sentence would exceed max_chunk_size, create a new chunk
            if len(current_chunk) + len(sentence) > self.max_chunk_size and current_chunk:
                chunk = Chunk(
                    text=current_chunk.strip(),
                    page_number=page_number,
                    document_id=document_id,
                    chunk_id=f"{document_id}_p{page_number}_c{chunk_count}",
                    metadata=metadata.copy()
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:] + sentence
                chunk_count += 1
            else:
                current_chunk += sentence
        
        # Add the last chunk if it's not empty
        if current_chunk.strip():
            chunk = Chunk(
                text=current_chunk.strip(),
                page_number=page_number,
                document_id=document_id,
                chunk_id=f"{document_id}_p{page_number}_c{chunk_count}",
                metadata=metadata.copy()
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for better chunking."""
        # Simple sentence splitting - can be improved with NLP libraries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Ensure each sentence ends with a space
        sentences = [s + " " if not s.endswith(" ") else s for s in sentences]
        return sentences

def process_documents(pdf_dir: str, output_dir: Optional[str] = None) -> List[Chunk]:
    """
    Process all PDF documents in a directory.
    
    Args:
        pdf_dir: Directory containing PDF files
        output_dir: Optional directory to save processed chunks
        
    Returns:
        List of all chunks from all documents
    """
    processor = DocumentProcessor()
    all_chunks = []
    
    # Ensure output directory exists if specified
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process each PDF file in the directory
    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, filename)
            document_id = os.path.splitext(filename)[0]
            
            # Process the PDF
            chunks = processor.process_pdf(pdf_path, document_id)
            all_chunks.extend(chunks)
            
            # Save chunks to output directory if specified
            if output_dir:
                import json
                
                output_file = os.path.join(output_dir, f"{document_id}_chunks.json")
                with open(output_file, 'w') as f:
                    json.dump([{
                        "text": chunk.text,
                        "page_number": chunk.page_number,
                        "document_id": chunk.document_id,
                        "chunk_id": chunk.chunk_id,
                        "metadata": chunk.metadata
                    } for chunk in chunks], f, indent=2)
                
                logger.info(f"Saved chunks to {output_file}")
    
    return all_chunks

if __name__ == "__main__":
    # Example usage
    pdf_dir = "C:\\Users\\logan\\Downloads\\RAG_test\\test_1"  # Double backslashes
    output_dir = "data/processed"
    
    chunks = process_documents(pdf_dir, output_dir)
    print(f"Processed {len(chunks)} chunks in total")
