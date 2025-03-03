# Simple page based chunking and RAG

A **Retrieval-Augmented Generation (RAG)** that processes PDFs, generates embeddings, and answers technical queries with precise page citations.

---

## Table of Contents

- [Introduction](#introduction)
- [System Architecture](#system-architecture)
- [Key Components](#key-components)
  - [Document Processing](#document-processing)
  - [Embedding Generation & Storage](#embedding-generation--storage)
  - [Retrieval & Response Generation](#retrieval--response-generation)
  - [API Endpoints](#api-endpoints)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
  - [Processing Documents](#processing-documents)
  - [Querying the System](#querying-the-system)
  - [API Usage](#api-usage)
- [Docker Deployment](#docker-deployment)
- [Future Enhancements](#future-enhancements)
- [License](#license)

---

## Introduction

This version of RAG leverages Azure OpenAI APIs and Supabase to transform PDF-based service manuals into a searchable knowledge base. The system extracts text from PDFs, chunks it with preserved page metadata, generates vector embeddings, and supports retrieval-augmented query responses with accurate page citations.

---

## System Architecture

The project is built on three major pillars:

1. **Document Processing Pipeline**  
   Extracts and cleans PDF content, then splits the text into page-anchored chunks.

2. **Embedding Generation and Storage**  
   Uses Azure OpenAI's `text-embedding-3-large` to create embeddings, which are stored in Supabase.

3. **Retrieval and Response Generation (RAG)**  
   Retrieves relevant document chunks and generates responses with citations using GPT-4o.

---
