# LLM-Powered-RAG-System

- [LLM-Powered-RAG-System](#llm-powered-rag-system)
  - [Frameworks](#frameworks)
  - [Projects](#projects)
  - [Components](#components)
    - [AI Agents](#ai-agents)
    - [Chat with Documents](#chat-with-documents)
    - [Database](#database)
    - [Optimize/Evaluation Method](#optimizeevaluation-method)
    - [Data Chunking](#data-chunking)
    - [Fine-tuning](#fine-tuning)
    - [Others](#others)
  - [Inference server](#inference-server)
  - [LLMs](#llms)
  - [Papers](#papers)
  - [Blog](#blog)
  - [Other Resources](#other-resources)

## Frameworks

- [langchain](https://github.com/langchain-ai/langchain) - ⚡ Building applications with LLMs through composability ⚡ ![GitHub Repo stars](https://img.shields.io/github/stars/langchain-ai/langchain?style=social)
- [llama_index](https://github.com/run-llama/llama_index) - LlamaIndex (formerly GPT Index) is a data framework for your LLM applications ![GitHub Repo stars](https://img.shields.io/github/stars/run-llama/llama_index?style=social)
- [crewAI](https://github.com/joaomdmoura/crewai/) - Framework for orchestrating role-playing, autonomous AI agents. By fostering collaborative intelligence, CrewAI empowers agents to work together seamlessly, tackling complex tasks. ![GitHub Repo stars](https://img.shields.io/github/stars/joaomdmoura/crewai?style=social)
- [embedchain](https://github.com/embedchain/embedchain) - Embedchain is an Open Source RAG Framework that makes it easy to create and deploy AI apps. - ![GitHub Repo stars](https://img.shields.io/github/stars/embedchain/embedchain?style=social)
- [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) - Dense Retrieval and Retrieval-augmented LLMs - ![GitHub Repo stars](https://img.shields.io/github/stars/FlagOpen/FlagEmbedding?style=social)
- [TaskingAI](https://github.com/TaskingAI/TaskingAI) - Dense Retrieval and Retrieval-augmented LLMs - ![GitHub Repo stars](https://img.shields.io/github/stars/TaskingAI/TaskingAI?style=social)
- [fastRAG](https://github.com/IntelLabs/fastRAG) - Efficient Retrieval Augmentation and Generation Framework - ![GitHub Repo stars](https://img.shields.io/github/stars/IntelLabs/fastRAG?style=social)
- [llmware](https://github.com/llmware-ai/llmware) - Providing enterprise-grade LLM-based development framework, tools, and fine-tuned models. - ![GitHub Repo stars](https://img.shields.io/github/stars/llmware-ai/llmware?style=social)
- [llm-applications](https://github.com/ray-project/llm-applications) - A comprehensive guide to building RAG-based LLM applications for production. - ![GitHub Repo stars](https://img.shields.io/github/stars/ray-project/llm-applications?style=social)
- [DB-GPT](https://github.com/eosphoros-ai/DB-GPT) - An open-source AI native data app development framework with AWEL(Agentic Workflow Expression Language) and agents. Revolutionizing Database Interactions with Private LLM Technology - ![GitHub Repo stars](https://img.shields.io/github/stars/eosphoros-ai/DB-GPT?style=social)
- [langroid](https://github.com/langroid/langroid) - Harness LLMs with Multi-Agent Programming - ![GitHub Repo stars](https://img.shields.io/github/stars/langroid/langroid?style=social)
- [pandas-ai](https://github.com/Sinaptik-AI/pandas-ai) - Chat with your data (SQL, CSV, pandas, polars, noSQL, etc). PandasAI makes data analysis conversational using LLMs (GPT 3.5 / 4, Anthropic, VertexAI) and RAG. - ![GitHub Repo stars](https://img.shields.io/github/stars/Sinaptik-AI/pandas-ai?style=social)
- [canopy](https://github.com/pinecone-io/canopy) - Retrieval Augmented Generation (RAG) framework and context engine powered by Pinecone - ![GitHub Repo stars](https://img.shields.io/github/stars/pinecone-io/canopy?style=social)
- [autollm](https://github.com/safevideo/autollm) - Ship RAG based LLM web apps in seconds. - ![GitHub Repo stars](https://img.shields.io/github/stars/safevideo/autollm?style=social)
- [GraphRAG](https://github.com/microsoft/graphrag) - A modular graph-based Retrieval-Augmented Generation (RAG) system. A data pipeline and transformation suite that is designed to extract meaningful, structured data from unstructured text using the power of LLMs. - ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/graphrag?style=social)


## Projects

- [Dify](https://github.com/langgenius/dify) - an open-source LLM app development platform. Dify's intuitive interface combines AI workflow, RAG pipeline, agent capabilities, model management, observability features and more, letting you quickly go from prototype to production.  - ![GitHub Repo stars](https://img.shields.io/github/stars/langgenius/dify?style=social)
- [quivr](https://github.com/StanGirard/quivr) - Your GenAI Second Brain 🧠 A personal productivity assistant (RAG) ⚡️🤖 Chat with your docs (PDF, CSV, ...) & apps using Langchain, GPT 3.5 / 4 turbo, Private, Anthropic, VertexAI, Ollama, LLMs, that you can share with users ! - Ship RAG based LLM web apps in seconds. - ![GitHub Repo stars](https://img.shields.io/github/stars/StanGirard/quivr?style=social)
- [Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat) - 基于 ChatGLM 等大语言模型与 Langchain 等应用框架实现，开源、可离线部署的 RAG 与 Agent 应用项目。. - ![GitHub Repo stars](https://img.shields.io/github/stars/chatchat-space/Langchain-Chatchat?style=social)
- [Chatbot UI](https://github.com/mckaywrigley/chatbot-ui) - The open-source AI chat app for everyone. - ![GitHub Repo stars](https://img.shields.io/github/stars/mckaywrigley/chatbot-ui?style=social)
- [Jan](https://github.com/janhq/jan) - Jan is an open-source alternative to ChatGPT that runs 100% offline on your computer. Multiple engine support (llama.cpp, TensorRT-LLM). - ![GitHub Repo stars](https://img.shields.io/github/stars/janhq/jan?style=social)
- [fastGPT](https://github.com/labring/FastGPT) - FastGPT is a knowledge-based platform built on the LLM, offers out-of-the-box data processing and model invocation capabilities, allows for workflow orchestration through Flow visualization! - ![GitHub Repo stars](https://img.shields.io/github/stars/labring/FastGPT?style=social)
- [LangChain-ChatGLM-Webui](https://github.com/X-D-Lab/LangChain-ChatGLM-Webui) - 基于LangChain和ChatGLM-6B等系列LLM的针对本地知识库的自动问答 - ![GitHub Repo stars](https://img.shields.io/github/stars/X-D-Lab/LangChain-ChatGLM-Webui?style=social)
- [anything-llm](https://github.com/Mintplex-Labs/anything-llm) - Open-source multi-user ChatGPT for all LLMs, embedders, and vector databases. Unlimited documents, messages, and users in one privacy-focused app. - ![GitHub Repo stars](https://img.shields.io/github/stars/Mintplex-Labs/anything-llm?style=social)
- [QAnything](https://github.com/netease-youdao/QAnything) - Question and Answer based on Anything.
- [danswer](https://github.com/danswer-ai/danswer) - Ask Questions in natural language and get Answers backed by private sources. Connects to tools like Slack, GitHub, Confluence, etc. - ![GitHub Repo stars](https://img.shields.io/github/stars/danswer-ai/danswer?style=social)
- [RAGFlow](https://github.com/infiniflow/ragflow) -  an open-source RAG (Retrieval-Augmented Generation) engine based on deep document understanding. - ![GitHub Repo stars](https://img.shields.io/github/stars/infiniflow/ragflow?style=social)
- [Bisheng](https://github.com/dataelement/bisheng) - an open LLM devops platform for next generation AI applications.. - ![GitHub Repo stars](https://img.shields.io/github/stars/dataelement/bisheng?style=social)
- [khoj](https://github.com/khoj-ai/khoj) - A copilot to search and chat (using RAG) with your knowledge base (pdf, markdown, org). Use powerful, online (e.g gpt4) or private, offline (e.g mistral) LLMs. - ![GitHub Repo stars](https://img.shields.io/github/stars/khoj-ai/khoj?style=social)
- [rags](https://github.com/run-llama/rags) - Build ChatGPT over your data, all with natural language - ![GitHub Repo stars](https://img.shields.io/github/stars/run-llama/rags?style=social)
- [Verba](https://github.com/weaviate/Verba) - Retrieval Augmented Generation (RAG) chatbot powered by Weaviate - ![GitHub Repo stars](https://img.shields.io/github/stars/weaviate/Verba?style=social)
- [llm-app](https://github.com/pathwaycom/llm-app) - LLM App templates for RAG, knowledge mining, and stream analytics. Ready to run with Docker,⚡in sync with your data sources. - ![GitHub Repo stars](https://img.shields.io/github/stars/pathwaycom/llm-app?style=social)
- [casibase](https://github.com/casibase/casibase) - ⚡️Open-source AI LangChain-like RAG (Retrieval-Augmented Generation) knowledge database with web UI and Enterprise SSO⚡️ - ![GitHub Repo stars](https://img.shields.io/github/stars/casibase/casibase?style=social)
- [trt-llm-rag-windows](https://github.com/NVIDIA/trt-llm-rag-windows) - A developer reference project for creating Retrieval Augmented Generation (RAG) chatbots on Windows using TensorRT-LLM - ![GitHub Repo stars](https://img.shields.io/github/stars/NVIDIA/trt-llm-rag-windows?style=social)
- [GPT-RAG](https://github.com/Azure/GPT-RAG) - GPT-RAG core is a Retrieval-Augmented Generation pattern running in Azure, using Azure Cognitive Search for retrieval and Azure OpenAI large language models to power ChatGPT-style and Q&A experiences. - ![GitHub Repo stars](https://img.shields.io/github/stars/Azure/GPT-RAG?style=social) 
- [rag-demystified](https://github.com/pchunduri6/rag-demystified) - An LLM-powered advanced RAG pipeline built from scratch - ![GitHub Repo stars](https://img.shields.io/github/stars/pchunduri6/rag-demystified?style=social)
- [LARS](https://github.com/abgulati/LARS) - An application for running LLMs locally on your device, with your documents, facilitating detailed citations in generated responses. - ![GitHub Repo stars](https://img.shields.io/github/stars/abgulati/LARS?style=social)

## Components

### AI Agents
- [AutoGroq](https://github.com/jgravelle/AutoGroq) - groundbreaking tool that revolutionizes the way users interact with AI assistants. By dynamically generating tailored teams of AI agents based on your project requirements. - ![GitHub Repo stars](https://img.shields.io/github/stars/jgravelle/AutoGroq?style=social) 💥 :boom:
- [micro-agent](https://github.com/BuilderIO/micro-agent) - An AI agent that writes (actually useful) code for you: JavaScript maily - ![GitHub Repo stars](https://img.shields.io/github/stars/BuilderIO/micro-agent?style=social) 💥 :boom:
  

### Chat with Documents

- [privateGPT](https://github.com/imartinez/privateGPT) - Interact with your documents using the power of GPT, 100% privately, no data leaks
- [localGPT](https://github.com/PromtEngineer/localGPT) - Chat with your documents on your local device using GPT models. No data leaves your device and 100% private.
- [ChatFiles](https://github.com/guangzhengli/ChatFiles) - Document Chatbot
- [pdfGPT](https://github.com/bhaskatripathi/pdfGPT) - PDF GPT allows you to chat with the contents of your PDF file by using GPT capabilities. The most effective open source solution to turn your pdf files in a chatbot!
- [chatd](https://github.com/BruceMacD/chatd) - Chat with your documents using local AI - ![GitHub Repo stars](https://img.shields.io/github/stars/BruceMacD/chatd?style=social)
- [IncarnaMind](https://github.com/junruxiong/IncarnaMind) - Connect and chat with your multiple documents (pdf and txt) through GPT 3.5, GPT-4 Turbo, Claude and Local Open-Source LLMs
- [ArXivChatGuru](https://github.com/RedisVentures/ArXivChatGuru) - Use ArXiv ChatGuru to talk to research papers. This app uses LangChain, OpenAI, Streamlit, and Redis as a vector database/semantic cache. - ![GitHub Repo stars](https://img.shields.io/github/stars/RedisVentures/ArXivChatGuru?style=social)
- [h2ogpt](https://github.com/h2oai/h2ogpt) - Private chat with local GPT with document, images, video, etc. - ![GitHub Repo stars](https://img.shields.io/github/stars/h2oai/h2ogpt?style=social)

### Database

- [qdrant](https://github.com/qdrant/qdrant) - High-performance, massive-scale Vector Database for the next generation of AI. Qdrant is also available as a fully managed Qdrant Cloud ⛅ including a free tier. - ![GitHub Repo stars](https://img.shields.io/github/stars/qdrant/qdrant?style=social)
- [vanna](https://github.com/vanna-ai/vanna) - 🤖 Chat with your SQL database 📊. Accurate Text-to-SQL Generation via LLMs using RAG 🔄. - ![GitHub Repo stars](https://img.shields.io/github/stars/vanna-ai/vanna?style=social)
- [txtai](https://github.com/neuml/txtai) - 💡 All-in-one open-source embeddings database for semantic search, LLM orchestration and language model workflows - ![GitHub Repo stars](https://img.shields.io/github/stars/neuml/txtai?style=social)
- [infinity](https://github.com/infiniflow/infinity) - The AI-native database built for LLM applications, providing incredibly fast vector and full-text search - ![GitHub Repo stars](https://img.shields.io/github/stars/infiniflow/infinity?style=social)
- [postgresml](https://github.com/postgresml/postgresml) - The GPU-powered AI application database. - ![GitHub Repo stars](https://img.shields.io/github/stars/postgresml/postgresml?style=social)
- [lancedb](https://github.com/lancedb/lancedb) - Developer-friendly, serverless vector database for AI applications. Easily add long-term memory to your LLM apps!
  

### Optimize/Evaluation Method

- [sparrow](https://github.com/katanaml/sparrow) - Data extraction with ML and LLM - ![GitHub Repo stars](https://img.shields.io/github/stars/katanaml/sparrow?style=social)
- [fastembed](https://github.com/qdrant/fastembed) - Fast, Accurate, Lightweight Python library to make State of the Art Embedding - ![GitHub Repo stars](https://img.shields.io/github/stars/qdrant/fastembed?style=social)
- [self-rag](https://github.com/AkariAsai/self-rag) - SELF-RAG: Learning to Retrieve, Generate and Critique through Self-reflection - ![GitHub Repo stars](https://img.shields.io/github/stars/AkariAsai/self-rag?style=social)
- [instructor](https://github.com/jxnl/instructor) - Your Gateway to Structured Outputs with OpenAI
- [swirl-search](https://github.com/swirlai/swirl-search) - Swirl is open source software that simultaneously searches multiple content sources and returns AI ranked results. - ![GitHub Repo stars](https://img.shields.io/github/stars/swirlai/swirl-search?style=social)
- [kernel-memory](https://github.com/microsoft/kernel-memory) - Index and query any data using LLM and natural language, tracking sources and showing citations. - ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/kernel-memory?style=social)
- [AutoRAG](https://github.com/Marker-Inc-Korea/AutoRAG) - a tool for finding optimal RAG pipeline for “your data.” 🔮 - ![GitHub Repo stars](https://img.shields.io/github/stars/Marker-Inc-Korea/AutoRAG?style=social)
- [promptfoo](https://github.com/promptfoo/promptfoo) - Test your prompts, agents, and RAGs. Use LLM evals to improve your app's quality and catch problems.  - ![GitHub Repo stars](https://img.shields.io/github/stars/promptfoo/promptfoo?style=social)
- [YiVal](https://github.com/YiVal/YiVal) - Your Automatic Prompt Engineering Assistant for GenAI Applications - ![GitHub Repo stars](https://img.shields.io/github/stars/YiVal/YiVal?style=social)

### Data Chunking

- [OmniParse](https://github.com/adithya-s-k/omniparse)  - OmniParse is a platform that ingests and parses any unstructured data into structured, actionable data optimized for GenAI (LLM) applications. Whether you are working with documents, tables, images, videos, audio files, or web pages, OmniParse prepares your data to be clean, structured, and ready for AI applications such as RAG, fine-tuning, and more ![GitHub Repo stars](https://img.shields.io/github/stars/adithya-s-k/omniparse?style=social)
- [Open Parse)](https://github.com/Filimoa/open-parse)  - Easily chunks complex documents the same way a human would, including Semantic Processing. ![GitHub Repo stars](https://img.shields.io/github/stars/Filimoa/open-parse?style=social)
- [Unstructured](https://github.com/Unstructured-IO/unstructured)  - open-source components for ingesting and pre-processing images and text documents, such as PDFs, HTML, Word docs, and many more.  ![GitHub Repo stars](https://img.shields.io/github/stars/Unstructured-IO/unstructured?style=social)
- [ExtractThinker](https://github.com/enoch3712/ExtractThinker) - Library to extract data from files and documents agnostically using LLMs. extract_thinker provides ORM-style interaction between files and LLMs, allowing for flexible and powerful document extraction workflows. - ![GitHub Repo stars](https://img.shields.io/github/stars/enoch3712/ExtractThinker?style=social)
- [nlm-ingestor](https://github.com/nlmatics/nlm-ingestor) - This repo provides the service code for llmsherpa API to parse PDF, HTML, DOCX, PPTX. - ![GitHub Repo stars](https://img.shields.io/github/stars/nlmatics/nlm-ingestor?style=social)
- [python-readability](https://github.com/buriy/python-readability) - Given an HTML document, extract and clean up the main body text and title. - ![GitHub Repo stars](https://img.shields.io/github/stars/buriy/python-readability?style=social)



### Fine-tuning

- [mistral-finetune](https://github.com/mistralai/mistral-finetune) - mistral-finetune is a light-weight codebase that enables memory-efficient and performant finetuning of Mistral's models.... - ![GitHub Repo stars](https://img.shields.io/github/stars/mistralai/mistral-finetune?style=social)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) - A WebUI for Efficient Fine-Tuning of 100+ LLMs (ACL 2024). - ![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/LLaMA-Factory?style=social)
- [axolotl](https://github.com/axolotl-ai-cloud/axolotl) - a tool designed to streamline the fine-tuning of various AI models, offering support for multiple configurations and architectures.. - ![GitHub Repo stars](https://img.shields.io/github/stars/axolotl-ai-cloud/axolotl?style=social)

  
### Others

- [chatgpt-retrieval-plugin](https://github.com/openai/chatgpt-retrieval-plugin) - The ChatGPT Retrieval Plugin lets you easily find personal or work documents by asking questions in natural language.
- [RAGxplorer](https://github.com/gabrielchua/RAGxplorer) - Open-source tool to visualise your RAG 🔮 - ![GitHub Repo stars](https://img.shields.io/github/stars/gabrielchua/RAGxplorer?style=social)
- [deep-chat](https://github.com/OvidijusParsiunas/deep-chat) - a fully customizable AI chat component that can be injected into your website with minimal to no effort. - ![GitHub Repo stars](https://img.shields.io/github/stars/OvidijusParsiunas/deep-chat?style=social)
- [Ollama-Laravel](https://github.com/cloudstudio/ollama-laravel) - a Laravel package providing seamless integration with the Ollama API.. - ![GitHub Repo stars](https://img.shields.io/github/stars/cloudstudio/ollama-laravel?style=social)
- [fabric](https://github.com/danielmiessler/fabric) - an open-source framework for augmenting humans using AI. It provides a modular framework for solving specific problems using a crowdsourced set of AI prompts that can be used anywhere. - ![GitHub Repo stars](https://img.shields.io/github/stars/danielmiessler/fabric?style=social)
- [n8n](https://github.com/n8n-io/n8n) - Free and source-available fair-code licensed workflow automation tool. Easily automate tasks across different services. - ![GitHub Repo stars](https://img.shields.io/github/stars/n8n-io/n8n?style=social)
- [Langtrace](https://github.com/Scale3-Labs/langtrace) - an open-source, Open Telemetry based end-to-end observability tool for LLM applications, providing real-time tracing, evaluations and metrics for popular LLMs, LLM frameworks, vectorDBs and more.. Integrate using Typescript, Python. - ![GitHub Repo stars](https://img.shields.io/github/stars/Scale3-Labs/langtrace?style=social)
- [tokencost](https://github.com/AgentOps-AI/tokencost) - Helps calculate the USD cost of using major Large Language Model (LLMs) APIs by calculating the estimated cost of prompts and completions. - ![GitHub Repo stars](https://img.shields.io/github/stars/AgentOps-AI/tokencost?style=social)
- [quality-prompts](https://github.com/sarthakrastogi/quality-prompts) - Use and evaluate prompting techniques quickly. - ![GitHub Repo stars](https://img.shields.io/github/stars/sarthakrastogi/quality-prompts?style=social)


## Inference server

- [mistral.rs](https://github.com/EricLBuehler/mistral.rs) - a fast LLM inference platform supporting inference on a variety of devices, quantization, and easy-to-use application with an Open-AI API compatible HTTP server and Python bindings. - ![GitHub Repo stars](https://img.shields.io/github/stars/EricLBuehler/mistral.rs?style=social)
- [MInference](https://github.com/microsoft/MInference) - To speed up Long-context LLMs' inference, approximate and dynamic sparse calculate the attention, which reduces inference latency by up to 10x for pre-filling on an A100 while maintaining accuracy. - ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/MInference?style=social)
- [LiteLLM](https://github.com/BerriAI/litellm) - Call all LLM APIs using the OpenAI format. Use Bedrock, Azure, OpenAI, Cohere, Anthropic, Ollama, Sagemaker, HuggingFace, Replicate (100+ LLMs). - ![GitHub Repo stars](https://img.shields.io/github/stars/BerriAI/litellm?style=social)


## LLMs

- [MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V) - a series of end-side multimodal LLMs (MLLMs) designed for vision-language understanding. Models take image and text as inputs and provide high-quality text outputs.- ![GitHub Repo stars](https://img.shields.io/github/stars/OpenBMB/MiniCPM-V?style=social)
- [DeepSeek-Coder-V2](https://github.com/deepseek-ai/DeepSeek-Coder-V2) - an open-source Mixture-of-Experts (MoE) code language model that achieves performance comparable to GPT4-Turbo in code-specific tasks. - ![GitHub Repo stars](https://img.shields.io/github/stars/deepseek-ai/DeepSeek-Coder-V2?style=social)
- [Trol](https://github.com/ByungKwanLee/TroL) - Traversal of Layers for Large Language and Vision Models. - ![GitHub Repo stars](https://img.shields.io/github/stars/ByungKwanLee/TroL?style=social)
- [FunAudioLLM](https://github.com/FunAudioLLM/FunAudioLLM-APP) - This project hosts two exciting applications leveraging advanced audio understand and speech generation models to bring your audio experiences to life: `CosyVoice` and `SenseVoice` - ![GitHub Repo stars](https://img.shields.io/github/stars/FunAudioLLM/FunAudioLLM-APP?style=social)
- [MobileLLM](https://github.com/facebookresearch/MobileLLM) - MobileLLM Optimizing Sub-billion Parameter Language Models for On-Device Use Cases. In ICML 2024. - ![GitHub Repo stars](https://img.shields.io/github/stars/facebookresearch/MobileLLM?style=social)
- [awesome-whisper](https://github.com/sindresorhus/awesome-whisper) - Awesome list for Whisper — an open-source AI-powered speech recognition system developed by OpenAI. - ![GitHub Repo stars](https://img.shields.io/github/stars/sindresorhus/awesome-whisper?style=social)


  
## Papers

- [Awesome-LLM-RAG](https://github.com/OpenBMB/MiniCPM-V) - This repo aims to record advanced papers of Retrieval Agumented Generation (RAG) in LLMs.


## Blog

- [Building RAG-based LLM Applications for Production](https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-1)
- [A-Guide-to-Retrieval-Augmented-LLM](https://github.com/Wang-Shuo/A-Guide-to-Retrieval-Augmented-LLM)


## Other Resources

- [funNLP](https://github.com/fighting41love/funNLP) - NLP民工的乐园: 几乎最全的中文NLP资源库, 在入门到熟悉NLP的过程中，用到了很多github上的包，遂整理了一下，分享在这里。
- [AGI-survey](https://github.com/ulab-uiuc/AGI-survey) - Awesome AGI Survey. Must-read papers on Artificial General Intelligence.
- [rag-resources](https://github.com/mrdbourke/rag-resources) - A collection of curated RAG (Retrieval Augmented Generation) resources.
- [RAG-Survey](https://github.com/Tongji-KGLLM/RAG-Survey)
- [Awesome-LLM-RAG-Application](https://github.com/lizhe2004/Awesome-LLM-RAG-Application) - the resources about the application based on LLM with RAG pattern
