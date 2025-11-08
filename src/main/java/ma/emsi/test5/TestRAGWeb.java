package ma.emsi.test5;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;
import ma.emsi.test1.Assistant;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class TestRAGWeb {

    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    private static EmbeddingStore<TextSegment> ingestDocument(
            Path documentPath,
            EmbeddingModel embeddingModel,
            DocumentParser parser,
            DocumentSplitter splitter) {

        System.out.println("  Ingestion : " + documentPath.getFileName());
        Document document = FileSystemDocumentLoader.loadDocument(documentPath, parser);
        List<TextSegment> segments = splitter.split(document);
        System.out.println("    - " + segments.size() + " segments crees");

        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);
        System.out.println("    - Embeddings stockes");

        return embeddingStore;
    }

    private static ContentRetriever createContentRetriever(
            EmbeddingStore<TextSegment> embeddingStore,
            EmbeddingModel embeddingModel) {

        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();
    }

    public static void main(String[] args) {
        configureLogger();
        System.out.println("=== Test 5 : RAG hybride (PDF + Web) ===\n");

        String geminiApiKey = System.getenv("GEMINI");
        if (geminiApiKey == null || geminiApiKey.isEmpty()) {
            System.err.println("Erreur : La variable d'environnement GEMINI n'est pas definie.");
            return;
        }

        String tavilyApiKey = System.getenv("Tvly");
        if (tavilyApiKey == null || tavilyApiKey.isEmpty()) {
            System.err.println("Erreur : La variable d'environnement TAVILY_API_KEY n'est pas definie.");
            System.err.println("Creez un compte sur https://tavily.com/ et definissez votre cle.");
            return;
        }

        System.out.println("=== Configuration du ChatModel ===\n");

        ChatModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(geminiApiKey)
                .modelName("gemini-2.0-flash-exp")
                .temperature(0.7)
                .logRequestsAndResponses(true)
                .build();
        System.out.println("ChatModel Gemini configure\n");

        System.out.println("=== PHASE 1 : Ingestion du document PDF ===\n");

        Path documentIA = Paths.get("src/main/resources/support_rag.pdf");
        DocumentParser parser = new ApacheTikaDocumentParser();
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        EmbeddingStore<TextSegment> embeddingStore = ingestDocument(
                documentIA, embeddingModel, parser, splitter);

        ContentRetriever pdfContentRetriever = createContentRetriever(embeddingStore, embeddingModel);
        System.out.println("ContentRetriever PDF cree\n");

        System.out.println("=== PHASE 2 : Configuration de la recherche Web ===\n");

        WebSearchEngine webSearchEngine = TavilyWebSearchEngine.builder()
                .apiKey(tavilyApiKey)
                .build();
        System.out.println("WebSearchEngine Tavily cree");

        ContentRetriever webContentRetriever = WebSearchContentRetriever.builder()
                .webSearchEngine(webSearchEngine)
                .build();
        System.out.println("ContentRetriever Web cree\n");

        System.out.println("=== PHASE 3 : Configuration du RAG hybride ===\n");

        QueryRouter queryRouter = new DefaultQueryRouter(pdfContentRetriever, webContentRetriever);
        System.out.println("QueryRouter cree avec 2 ContentRetrievers");

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();
        System.out.println("RetrievalAugmentor cree\n");

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(retrievalAugmentor)
                .build();
        System.out.println("Assistant RAG hybride pret\n");

        System.out.println("=".repeat(80));
        System.out.println("Le systeme combine automatiquement :");
        System.out.println("  1. Informations du document PDF local");
        System.out.println("  2. Informations recherchees sur le Web via Tavily");
        System.out.println("\nTapez 'quitter' ou 'exit' pour terminer.");
        System.out.println("=".repeat(80) + "\n");

        Scanner scanner = new Scanner(System.in);

        while (true) {
            System.out.print("Votre question : ");
            String question = scanner.nextLine().trim();

            if (question.equalsIgnoreCase("quitter") || question.equalsIgnoreCase("exit")) {
                System.out.println("\nAu revoir !");
                break;
            }

            if (question.isEmpty()) {
                System.out.println("Veuillez poser une question.\n");
                continue;
            }

            try {
                System.out.println("\n[Recherche dans le PDF ET sur le Web...]");
                System.out.println("=".repeat(80));
                String reponse = assistant.chat(question);
                System.out.println("=".repeat(80));
                System.out.println("\n--- Reponse ---");
                System.out.println(reponse);
                System.out.println("---------------\n");
            } catch (Exception e) {
                System.err.println("Erreur : " + e.getMessage());
                e.printStackTrace();
                System.out.println();
            }
        }
        scanner.close();
    }
}