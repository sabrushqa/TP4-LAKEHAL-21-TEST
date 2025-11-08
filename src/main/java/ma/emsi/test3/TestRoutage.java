package ma.emsi.test3;

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
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import ma.emsi.test1.Assistant;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class TestRoutage {

    /**
     * Configure le logger pour voir les détails du routage
     */
    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    /**
     * Ingère un document : charge, découpe, crée les embeddings et les stocke
     */
    private static EmbeddingStore<TextSegment> ingestDocument(
            Path documentPath,
            EmbeddingModel embeddingModel,
            DocumentParser parser,
            DocumentSplitter splitter) {

        System.out.println("Ingestion du document : " + documentPath.getFileName());
        Document document = FileSystemDocumentLoader.loadDocument(documentPath, parser);
        List<TextSegment> segments = splitter.split(document);
        System.out.println("  - " + segments.size() + " segments créés");

        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);
        System.out.println("  - Embeddings stockés\n");

        return embeddingStore;
    }

    /**
     * Crée un ContentRetriever pour un EmbeddingStore donné
     */
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
        // Configure le logging pour voir le routage
        configureLogger();
        System.out.println("=== Test 3 : Routage entre plusieurs sources ===\n");

        String geminiApiKey = System.getenv("GEMINI");
        if (geminiApiKey == null || geminiApiKey.isEmpty()) {
            System.err.println("Erreur : La variable d'environnement GEMINI n'est pas définie.");
            return;
        }

        // Création du ChatModel (version 1.8.0)
        System.out.println("Connexion au modèle Gemini...");
        ChatModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(geminiApiKey)
                .modelName("gemini-2.0-flash-exp")
                .temperature(0.7)
                .logRequestsAndResponses(true)
                .build();
        System.out.println("Modèle connecté\n");

        // Chemins des deux documents
        Path documentIA = Paths.get("src/main/resources/support_rag.pdf");
        Path documentAutre = Paths.get("src/main/resources/MobileAI-2.pdf");

        System.out.println("=== PHASE 1 : Ingestion des documents ===\n");

        // Initialisation commune
        DocumentParser parser = new ApacheTikaDocumentParser();
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // Ingestion des deux documents
        EmbeddingStore<TextSegment> embeddingStore1 = ingestDocument(
                documentIA, embeddingModel, parser, splitter);
        EmbeddingStore<TextSegment> embeddingStore2 = ingestDocument(
                documentAutre, embeddingModel, parser, splitter);

        System.out.println("=== PHASE 2 : Configuration du routage ===\n");

        // Création des ContentRetrievers
        ContentRetriever retriever1 = createContentRetriever(embeddingStore1, embeddingModel);
        ContentRetriever retriever2 = createContentRetriever(embeddingStore2, embeddingModel);

        // Description des sources pour le QueryRouter
        Map<ContentRetriever, String> retrieverDescriptions = new HashMap<>();
        retrieverDescriptions.put(retriever1,
                "Documents techniques sur l'intelligence artificielle, le RAG (Retrieval-Augmented Generation), " +
                        "LangChain4j, les modèles de langage (LLM), les embeddings, les techniques avancées de RAG, " +
                        "le machine learning et les réseaux de neurones");
        retrieverDescriptions.put(retriever2,
                "Documents sur le développement d'applications mobiles, Android, Kotlin, " +
                        "les coroutines, les Flows, Room database, et l'architecture des applications mobiles");

        System.out.println("Sources configurées :");
        System.out.println("  1. Document IA : " + documentIA.getFileName());
        System.out.println("  2. Document Mobile : " + documentAutre.getFileName());
        System.out.println();

        // Création du QueryRouter qui utilisera le LLM pour choisir la bonne source
        QueryRouter queryRouter = new LanguageModelQueryRouter(chatModel, retrieverDescriptions);

        // Création du RetrievalAugmentor avec le QueryRouter
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        // Création de l'assistant (version 1.8.0 utilise .chatModel())
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel) // Version 1.8.0
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        System.out.println("=== Assistant RAG avec Routage prêt ! ===\n");
        System.out.println("Le LLM choisira automatiquement la bonne source selon votre question.");
        System.out.println("Consultez les logs pour voir le processus de décision.\n");

        // Boucle de questions-réponses
        Scanner scanner = new Scanner(System.in);
        System.out.println("Tapez 'quitter' pour arrêter\n");

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
                System.out.println("\n[Analyse de la question et routage...]");
                System.out.println("=".repeat(70));
                String reponse = assistant.chat(question);
                System.out.println("=".repeat(70));
                System.out.println("\n--- Réponse ---");
                System.out.println(reponse);
                System.out.println("---------------\n");
            } catch (Exception e) {
                System.err.println("Erreur : " + e.getMessage());
                e.printStackTrace();
            }
        }
        scanner.close();
    }
}